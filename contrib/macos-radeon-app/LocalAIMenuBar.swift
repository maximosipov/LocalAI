// LocalAI M — menu-bar app for the Vulkan/MoltenVK LocalAI build on a 2019 MacBook Pro.
//
// Runs as an accessory app (menu bar only, no Dock icon). It owns the local-ai server
// process: starts it with the MoltenVK environment and the Radeon pinned, polls /readyz
// to drive the status display, and stops it on quit.
//
// Built by contrib/macos-radeon-app/build-app.sh into Contents/MacOS/LocalAI.

import AppKit
import Foundation

final class Controller: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var server: Process?
    private var pollTimer: Timer?
    private var ready = false

    private let addr = ProcessInfo.processInfo.environment["LOCALAI_ADDR"] ?? "127.0.0.1:8085"

    private var resources: URL { Bundle.main.resourceURL! }
    private var frameworks: URL {
        Bundle.main.bundleURL.appendingPathComponent("Contents/Frameworks")
    }
    private var support: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LocalAI")
    }
    private var modelsDir: URL { support.appendingPathComponent("models") }
    private var logURL: URL { support.appendingPathComponent("logs/local-ai.log") }

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "cpu", accessibilityDescription: "LocalAI")
            button.image?.isTemplate = true
            if button.image == nil { button.title = "AI" }
        }
        rebuildMenu()
        prepareSupportDirectory()
        startServer()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.pollReady()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        stopServer()
    }

    // MARK: - Support directory

    /// Seed model configs on first run. Weights are never bundled (they are gigabytes);
    /// if a development checkout is present its weights are symlinked in.
    private func prepareSupportDirectory() {
        let fm = FileManager.default
        try? fm.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        try? fm.createDirectory(at: support.appendingPathComponent("logs"),
                                withIntermediateDirectories: true)

        let seed = resources.appendingPathComponent("models-seed")
        for yaml in (try? fm.contentsOfDirectory(at: seed, includingPropertiesForKeys: nil)) ?? [] {
            let dst = modelsDir.appendingPathComponent(yaml.lastPathComponent)
            if !fm.fileExists(atPath: dst.path) { try? fm.copyItem(at: yaml, to: dst) }
        }

        // Weights are not bundled. If a development checkout is present, symlink its
        // weights into the models folder; LOCALAI_DEV_MODELS points at another one.
        let devModels = ProcessInfo.processInfo.environment["LOCALAI_DEV_MODELS"]
            .map { URL(fileURLWithPath: $0) }
            ?? fm.homeDirectoryForCurrentUser
                .appendingPathComponent("Projects/macbook-pro-2019-local-ai/scratchpad/models-localai")
        guard fm.fileExists(atPath: devModels.path) else { return }
        for f in (try? fm.contentsOfDirectory(at: devModels, includingPropertiesForKeys: nil)) ?? [] {
            guard ["gguf", "safetensors"].contains(f.pathExtension) else { continue }
            let dst = modelsDir.appendingPathComponent(f.lastPathComponent)
            guard !fm.fileExists(atPath: dst.path) else { continue }
            let target = (try? fm.destinationOfSymbolicLink(atPath: f.path)) ?? f.path
            try? fm.createSymbolicLink(atPath: dst.path, withDestinationPath: target)
        }
    }

    /// Backends are spawned by LocalAI as child processes, so the MoltenVK environment has
    /// to travel with them. A wrapper script per backend is the reliable way to guarantee it.
    private func writeWrapper(name: String, exe: URL, extraEnv: [String: String]) -> URL {
        let icd = resources.appendingPathComponent("vulkan/MoltenVK_icd.json").path
        var lines = [
            "#!/usr/bin/env bash",
            "export VK_ICD_FILENAMES=\"\(icd)\"",
            "export DYLD_LIBRARY_PATH=\"\(frameworks.path):${DYLD_LIBRARY_PATH:-}\"",
            // Pin the discrete Radeon. The integrated GPU is never used.
            "export GGML_VK_VISIBLE_DEVICES=0",
        ]
        for (k, v) in extraEnv.sorted(by: { $0.key < $1.key }) {
            lines.append("export \(k)=\"\(v)\"")
        }
        lines.append("exec \"\(exe.path)\" \"$@\"")
        let url = support.appendingPathComponent(name)
        try? lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
        try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    // MARK: - Server

    private func startServer() {
        let fm = FileManager.default
        let llamaExe = resources.appendingPathComponent("backends/llama-cpp/grpc-server")
        var external = ["llama-cpp:\(writeWrapper(name: "run-llama-vulkan.sh", exe: llamaExe, extraEnv: [:]).path)"]

        let sdExe = resources.appendingPathComponent("backends/stablediffusion-ggml/stablediffusion-ggml")
        if fm.fileExists(atPath: sdExe.path) {
            let dylib = resources.appendingPathComponent("backends/stablediffusion-ggml/libgosd-fallback.dylib")
            let so = resources.appendingPathComponent("backends/stablediffusion-ggml/libgosd-fallback.so")
            let lib = fm.fileExists(atPath: dylib.path) ? dylib : so
            let wrapper = writeWrapper(name: "run-sd-vulkan.sh", exe: sdExe,
                                       extraEnv: ["SD_LIBRARY": lib.path])
            external.append("stablediffusion-ggml:\(wrapper.path)")
        }

        // LocalAI resolves its data, backends and configuration directories relative to
        // the working directory. An app bundle launches with cwd=/, which turns the data
        // path into "//data" on a read-only filesystem and the server exits at startup.
        // Pass all of them explicitly and keep them beside the models.
        let dataDir = support.appendingPathComponent("data")
        let backendsDir = support.appendingPathComponent("backends")
        let configDir = support.appendingPathComponent("configuration")
        let generatedDir = support.appendingPathComponent("generated")
        for dir in [dataDir, backendsDir, configDir, generatedDir] {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        let p = Process()
        p.executableURL = resources.appendingPathComponent("local-ai")
        p.currentDirectoryURL = support
        p.arguments = ["run", "--address", addr, "--models-path", modelsDir.path,
                       "--backends-path", backendsDir.path,
                       "--localai-config-dir", configDir.path,
                       "--generated-content-path", generatedDir.path,
                       // v4.12's add-ons need a runtime make build does not produce;
                       // without this the server exits at startup.
                       "--addons-enabled=false",
                       "--external-grpc-backends", external.joined(separator: ",")]
        var env = ProcessInfo.processInfo.environment
        env["LOCALAI_DATA_PATH"] = dataDir.path
        env["VK_ICD_FILENAMES"] = resources.appendingPathComponent("vulkan/MoltenVK_icd.json").path
        env["DYLD_LIBRARY_PATH"] = frameworks.path
        env["GGML_VK_VISIBLE_DEVICES"] = "0"
        // LocalAI's macOS hardware probe reports VRAM 0 and can zero gpu_layers.
        env["LOCALAI_DISABLE_HARDWARE_DEFAULTS"] = "true"
        p.environment = env

        if !fm.fileExists(atPath: logURL.path) { fm.createFile(atPath: logURL.path, contents: nil) }
        if let handle = try? FileHandle(forWritingTo: logURL) {
            handle.seekToEndOfFile()
            p.standardOutput = handle
            p.standardError = handle
        }
        do { try p.run(); server = p } catch { notify("LocalAI failed to start: \(error.localizedDescription)") }
        rebuildMenu()
    }

    private func stopServer() {
        guard let p = server, p.isRunning else { return }
        p.terminate()
        // Backends are separate child processes; give them a moment to follow the parent.
        let deadline = Date().addingTimeInterval(5)
        while p.isRunning && Date() < deadline { usleep(100_000) }
        if p.isRunning { kill(p.processIdentifier, SIGKILL) }
        server = nil
        ready = false
    }

    @objc private func restartServer() {
        stopServer()
        startServer()
    }

    private func pollReady() {
        guard let url = URL(string: "http://\(addr)/readyz") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 1.5
        URLSession.shared.dataTask(with: request) { [weak self] _, response, _ in
            let ok = (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async {
                guard let self, self.ready != ok else { return }
                self.ready = ok
                self.rebuildMenu()
            }
        }.resume()
    }

    // MARK: - Menu

    private func rebuildMenu() {
        let menu = NSMenu()
        let running = server?.isRunning ?? false
        let state = !running ? "stopped" : (ready ? "ready — http://\(addr)" : "starting…")
        let header = NSMenuItem(title: "LocalAI M: \(state)", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        menu.addItem(.separator())

        let webui = NSMenuItem(title: "Open WebUI", action: #selector(openWebUI), keyEquivalent: "o")
        webui.target = self
        webui.isEnabled = ready
        menu.addItem(webui)

        let copy = NSMenuItem(title: "Copy API Base URL", action: #selector(copyURL), keyEquivalent: "c")
        copy.target = self
        menu.addItem(copy)
        menu.addItem(.separator())

        for (title, sel) in [("Open Models Folder", #selector(openModels)),
                             ("Show Log", #selector(showLog)),
                             ("Restart Server", #selector(restartServer))] {
            let item = NSMenuItem(title: title, action: sel, keyEquivalent: "")
            item.target = self
            menu.addItem(item)
        }
        menu.addItem(.separator())

        let quit = NSMenuItem(title: "Quit LocalAI M", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
        statusItem.menu = menu
    }

    @objc private func openWebUI() { NSWorkspace.shared.open(URL(string: "http://\(addr)/")!) }
    @objc private func openModels() { NSWorkspace.shared.open(modelsDir) }
    @objc private func showLog() { NSWorkspace.shared.selectFile(logURL.path, inFileViewerRootedAtPath: support.path) }

    @objc private func copyURL() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("http://\(addr)/v1", forType: .string)
    }

    @objc private func quit() {
        stopServer()
        NSApp.terminate(nil)
    }

    private func notify(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "LocalAI M"
        alert.informativeText = message
        alert.runModal()
    }
}

let app = NSApplication.shared
let controller = Controller()
app.delegate = controller
app.setActivationPolicy(.accessory)
app.run()
