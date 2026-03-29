//
//  ContentView.swift
//  ScatteringCam
//
//  Created by Peter-Bernd Otte on 28.03.26.
//

@preconcurrency import AVFoundation
import Combine
import CoreML
import Darwin
import SwiftUI
import UIKit
import Vision

struct ContentView: View {
    @StateObject private var camera = CameraViewModel()
    @State private var isSettingsPresented = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                cameraPreview
                debugPanel
            }
            .navigationTitle("ScatteringCam")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        isSettingsPresented = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                    .accessibilityLabel("Einstellungen")
                }
            }
        }
        .task {
            await camera.start()
        }
        .onChange(of: isSettingsPresented) { _, isPresented in
            if isPresented {
                camera.pauseForSettings()
            } else {
                camera.resumeAfterSettings()
            }
        }
        .background(Color.black)
        .fullScreenCover(isPresented: $isSettingsPresented) {
            SettingsView(
                confidenceThreshold: Binding(
                    get: { camera.confidenceThreshold },
                    set: { camera.updateConfidenceThreshold($0) }
                ),
                maxMissedFramesForTracking: Binding(
                    get: { camera.maxMissedFramesForTracking },
                    set: { camera.updateMaxMissedFramesForTracking($0) }
                ),
                trailDuration: Binding(
                    get: { camera.trailDuration },
                    set: { camera.updateTrailDuration($0) }
                ),
                centerCropEnabled: Binding(
                    get: { camera.centerCropEnabled },
                    set: { camera.updateCenterCropEnabled($0) }
                ),
                selectedCamera: Binding(
                    get: { camera.selectedCamera },
                    set: { camera.updateSelectedCamera($0) }
                ),
                cameraFPS: Binding(
                    get: { camera.cameraFPS },
                    set: { camera.updateCameraFPS($0) }
                )
            )
        }
    }

    private var cameraPreview: some View {
        GeometryReader { geometry in
            let sideLength = geometry.size.width

            ZStack {
                Color.black

                ZStack {
                    CameraPreviewView(
                        session: camera.captureSession,
                        videoGravity: camera.previewVideoGravity
                    )
                    .frame(width: sideLength, height: sideLength)

                    DetectionOverlay(
                        detections: camera.detections,
                        trails: camera.trails,
                        imageSize: camera.imageSize
                    )
                    .frame(width: sideLength, height: sideLength)
                }
                .frame(width: sideLength, height: sideLength)
            }
            .frame(width: sideLength, height: sideLength)
            .clipped()
        }
        .frame(maxWidth: .infinity)
        .aspectRatio(1, contentMode: .fit)
        .frame(maxWidth: .infinity)
    }

    private var debugPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("Model Debug")
                    .font(.headline)

                Text(camera.statusText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                if let errorMessage = camera.errorMessage {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                Divider()

                debugLine("Modell", camera.debugInfo.modelName)
                debugLine("Modell geladen", camera.debugInfo.isModelLoaded ? "ja" : "nein")
                debugLine("Kamera", camera.debugInfo.cameraTitle)
                debugLine("Kamera FPS", camera.debugInfo.cameraFPSText)
                debugLine("Bildgroesse", camera.debugInfo.imageSizeText)
                debugLine("Center Crop", camera.debugInfo.centerCropEnabled ? "ja" : "nein")
                debugLine("Model FPS", "\(camera.debugInfo.modelFPSText) FPS")
                debugLine("RAM", camera.debugInfo.memoryUsageText)
                debugLine("CPU", camera.debugInfo.cpuUsageText)
                debugLine("Aktive Tracks", "\(camera.debugInfo.activeTrackCount)")
                debugLine("Roh-Ergebnisse", "\(camera.debugInfo.rawObservationCount)")
                debugLine("Gefiltert", "\(camera.debugInfo.filteredDetectionCount)")
                debugLine("Bester Treffer", camera.debugInfo.topPredictionText)
                debugLine("Output", camera.debugInfo.outputSummary)

                if !camera.detections.isEmpty {
                    debugSection(title: "Treffer", rows: camera.formattedDetections)
                }

                if !camera.debugInfo.rawPredictions.isEmpty {
                    debugSection(title: "Top Vorhersagen", rows: camera.debugInfo.rawPredictions)
                }

                if !camera.debugInfo.rawRows.isEmpty {
                    debugSection(title: "Rohzeilen", rows: camera.debugInfo.rawRows)
                }

            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }

    private func debugLine(_ title: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(.caption.weight(.semibold))
            Spacer()
            Text(value)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
        }
    }

    private func debugSection(title: String, rows: [String]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption.weight(.semibold))
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                Text(row)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
    }
}

@MainActor
final class CameraViewModel: ObservableObject {
    @Published private(set) var detections: [Detection] = []
    @Published private(set) var statusText = "Initialisiere Kamera..."
    @Published private(set) var errorMessage: String?
    @Published private(set) var imageSize: CGSize = .zero
    @Published private(set) var confidenceThreshold: Double = 0.35
    @Published private(set) var maxMissedFramesForTracking = defaultTrackToleranceValue
    @Published private(set) var trailDuration: TrailDurationOption = .off
    @Published private(set) var centerCropEnabled = true
    @Published private(set) var selectedCamera: CameraOption = .back
    @Published private(set) var cameraFPS: CameraFPSOption = .fps30
    @Published private(set) var trails: [TrackTrail] = []
    @Published private(set) var debugInfo = DebugInfo(
        modelName: "best",
        isModelLoaded: false
    )

    private let captureController = CameraCaptureController(
        modelName: "best",
        confidenceThreshold: 0.35
    )

    init() {
        captureController.delegate = self
        debugInfo.confidenceThreshold = confidenceThreshold
        debugInfo.maxMissedFramesForTracking = maxMissedFramesForTracking
        debugInfo.trailDuration = trailDuration
        debugInfo.centerCropEnabled = centerCropEnabled
        debugInfo.selectedCamera = selectedCamera
        debugInfo.cameraFPS = cameraFPS
        captureController.updateCenterCropEnabled(centerCropEnabled)
        captureController.updateMaxMissedFramesForTracking(maxMissedFramesForTracking)
        captureController.updateTrailDuration(trailDuration.duration)
        captureController.updateSelectedCamera(selectedCamera)
        captureController.updateCameraFPS(cameraFPS)
    }

    var captureSession: AVCaptureSession {
        captureController.session
    }

    var previewVideoGravity: AVLayerVideoGravity {
        centerCropEnabled ? .resizeAspectFill : .resize
    }

    var formattedDetections: [String] {
        detections.enumerated().map { index, detection in
            let box = detection.boundingBox
            let trackPrefix = detection.trackID.map { "#\($0) " } ?? ""
            return "t\(index): \(trackPrefix)\(detection.label) \(Int(detection.confidence * 100))% x \(Int(box.minX * 640)) y \(Int(box.minY * 640)) w \(Int(box.width * 640)) h \(Int(box.height * 640))"
        }
    }

    func updateConfidenceThreshold(_ value: Double) {
        let steppedValue = (value / 0.05).rounded() * 0.05
        confidenceThreshold = steppedValue
        captureController.updateConfidenceThreshold(Float(steppedValue))
        debugInfo.confidenceThreshold = steppedValue
    }

    func updateCenterCropEnabled(_ isEnabled: Bool) {
        centerCropEnabled = isEnabled
        captureController.updateCenterCropEnabled(isEnabled)
        debugInfo.centerCropEnabled = isEnabled
    }

    func updateMaxMissedFramesForTracking(_ value: Int) {
        let snappedValue = snappedTrackToleranceValue(value)
        maxMissedFramesForTracking = snappedValue
        captureController.updateMaxMissedFramesForTracking(snappedValue)
        debugInfo.maxMissedFramesForTracking = snappedValue
    }

    func updateTrailDuration(_ option: TrailDurationOption) {
        trailDuration = option
        if option == .off {
            trails = []
        }
        captureController.updateTrailDuration(option.duration)
        debugInfo.trailDuration = option
    }

    func updateSelectedCamera(_ cameraOption: CameraOption) {
        guard selectedCamera != cameraOption else { return }
        selectedCamera = cameraOption
        debugInfo.selectedCamera = cameraOption
        captureController.updateSelectedCamera(cameraOption)
    }

    func updateCameraFPS(_ option: CameraFPSOption) {
        guard cameraFPS != option else { return }
        cameraFPS = option
        debugInfo.cameraFPS = option
        captureController.updateCameraFPS(option)
    }

    func pauseForSettings() {
        captureController.stopRunning()
        statusText = "Kamera pausiert"
    }

    func resumeAfterSettings() {
        guard errorMessage == nil || AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
            return
        }
        captureController.startRunning()
        statusText = "Kamera aktiv. Live-Bild wird analysiert."
    }

    func start() async {
        let authorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)

        switch authorizationStatus {
        case .authorized:
            await configureAndRunSessionIfNeeded()
        case .notDetermined:
            statusText = "Frage Kamerazugriff an..."
            let isGranted = await requestVideoAccess()
            guard isGranted else {
                statusText = "Kein Kamerazugriff"
                errorMessage = "Erlaube den Kamerazugriff in den iPhone-Einstellungen."
                return
            }
            await configureAndRunSessionIfNeeded()
        default:
            statusText = "Kein Kamerazugriff"
            errorMessage = "Die Kamera ist deaktiviert. Aktiviere sie in Einstellungen > Datenschutz > Kamera."
        }
    }

    private func configureAndRunSessionIfNeeded() async {
        do {
            try await captureController.configureIfNeeded()
            captureController.startRunning()
        } catch {
            statusText = "Initialisierung fehlgeschlagen"
            errorMessage = error.localizedDescription
        }
    }

    private func requestVideoAccess() async -> Bool {
        await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .video) { granted in
                continuation.resume(returning: granted)
            }
        }
    }
}

@MainActor
protocol CameraCaptureControllerDelegate: AnyObject {
    func cameraCaptureController(
        _ controller: CameraCaptureController,
        didUpdateDetections detections: [Detection],
        trails: [TrackTrail],
        imageSize: CGSize
    )

    func cameraCaptureController(
        _ controller: CameraCaptureController,
        didUpdateStatus statusText: String,
        errorMessage: String?
    )

    func cameraCaptureController(
        _ controller: CameraCaptureController,
        didUpdateDebugInfo debugInfo: DebugInfo
    )
}

final class CameraCaptureController: NSObject, @unchecked Sendable {
    weak var delegate: CameraCaptureControllerDelegate?

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let videoOutputQueue = DispatchQueue(label: "camera.video.queue")
    private let modelName: String
    private let videoOutput = AVCaptureVideoDataOutput()
    private lazy var sampleBufferDelegate = VideoOutputDelegate { [weak self] sampleBuffer in
        self?.processSampleBuffer(sampleBuffer)
    }

    private var coreMLModel: MLModel?
    private var visionModel: VNCoreMLModel?
    private var confidenceThreshold: Float
    private var modelInputSize = CGSize(width: 640, height: 640)
    private var didConfigureSession = false
    private var isProcessingFrame = false
    private let frameOrientation = CGImagePropertyOrientation.up
    private var lastInferenceTimestamp: CFTimeInterval?
    private var smoothedFPS: Double = 0
    private var centerCropEnabled = true
    private var selectedCamera: CameraOption = .back
    private var cameraFPS: CameraFPSOption = .fps30
    private var maxMissedFramesForTracking = defaultTrackToleranceValue
    private var trailDuration: TimeInterval = 0
    private var currentCameraInput: AVCaptureDeviceInput?
    private var trackedObjects: [TrackedObject] = []
    private var nextTrackID = 1

    init(modelName: String, confidenceThreshold: Float) {
        self.modelName = modelName
        self.confidenceThreshold = confidenceThreshold
    }

    func updateConfidenceThreshold(_ value: Float) {
        confidenceThreshold = value
    }

    func updateCenterCropEnabled(_ isEnabled: Bool) {
        centerCropEnabled = isEnabled
    }

    func updateMaxMissedFramesForTracking(_ value: Int) {
        maxMissedFramesForTracking = max(value, 0)
    }

    func updateTrailDuration(_ value: TimeInterval) {
        trailDuration = max(value, 0)
        trimTrackHistories(referenceTime: CACurrentMediaTime())
    }

    func updateSelectedCamera(_ cameraOption: CameraOption) {
        selectedCamera = cameraOption

        Task { @MainActor in
            delegate?.cameraCaptureController(
                self,
                didUpdateDebugInfo: DebugInfo(
                    modelName: self.modelName,
                    isModelLoaded: self.coreMLModel != nil,
                    confidenceThreshold: Double(self.currentConfidenceThreshold),
                    centerCropEnabled: self.currentCenterCropEnabled,
                    selectedCamera: cameraOption,
                    cameraFPS: self.currentCameraFPS,
                    maxMissedFramesForTracking: self.currentMaxMissedFramesForTracking,
                    trailDuration: self.currentTrailDurationOption
                )
            )
        }

        guard didConfigureSession else { return }

        sessionQueue.async {
            do {
                try self.replaceCameraInputIfNeeded()
                Task { @MainActor in
                    self.delegate?.cameraCaptureController(
                        self,
                        didUpdateStatus: "Kamera aktiv. Live-Bild wird analysiert.",
                        errorMessage: nil
                    )
                }
            } catch {
                Task { @MainActor in
                    self.delegate?.cameraCaptureController(
                        self,
                        didUpdateStatus: "Kamerawechsel fehlgeschlagen",
                        errorMessage: error.localizedDescription
                    )
                }
            }
        }
    }

    func updateCameraFPS(_ option: CameraFPSOption) {
        cameraFPS = option

        Task { @MainActor in
            delegate?.cameraCaptureController(
                self,
                didUpdateDebugInfo: DebugInfo(
                    modelName: self.modelName,
                    isModelLoaded: self.coreMLModel != nil,
                    confidenceThreshold: Double(self.currentConfidenceThreshold),
                    centerCropEnabled: self.currentCenterCropEnabled,
                    selectedCamera: self.currentSelectedCamera,
                    cameraFPS: option,
                    maxMissedFramesForTracking: self.currentMaxMissedFramesForTracking,
                    trailDuration: self.currentTrailDurationOption
                )
            )
        }

        guard didConfigureSession else { return }

        sessionQueue.async {
            do {
                try self.applyPreferredFrameRate()
                Task { @MainActor in
                    self.delegate?.cameraCaptureController(
                        self,
                        didUpdateStatus: "Kamera aktiv. Live-Bild wird analysiert.",
                        errorMessage: nil
                    )
                }
            } catch {
                Task { @MainActor in
                    self.delegate?.cameraCaptureController(
                        self,
                        didUpdateStatus: "FPS-Aenderung fehlgeschlagen",
                        errorMessage: error.localizedDescription
                    )
                }
            }
        }
    }

    func configureIfNeeded() async throws {
        guard !didConfigureSession else { return }

        try loadModel()
        let isModelLoaded = coreMLModel != nil
        let statusText = isModelLoaded
            ? "Kamera aktiv. Live-Bild wird analysiert."
            : "Lege dein kompiliertes CoreML-Modell '\(modelName)' ins App-Bundle."

        Task { @MainActor in
            delegate?.cameraCaptureController(self, didUpdateStatus: statusText, errorMessage: isModelLoaded ? nil : "Kein Modell gefunden. Füge \(modelName).mlmodel oder \(modelName).mlpackage zum Xcode-Projekt hinzu.")
            delegate?.cameraCaptureController(
                self,
                didUpdateDebugInfo: DebugInfo(
                    modelName: self.modelName,
                    isModelLoaded: isModelLoaded,
                    confidenceThreshold: Double(self.currentConfidenceThreshold),
                    centerCropEnabled: self.currentCenterCropEnabled,
                    selectedCamera: self.currentSelectedCamera,
                    cameraFPS: self.currentCameraFPS,
                    maxMissedFramesForTracking: self.currentMaxMissedFramesForTracking,
                    trailDuration: self.currentTrailDurationOption
                )
            )
        }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do {
                    self.session.beginConfiguration()
                    self.session.sessionPreset = .hd1280x720

                    try self.configureCameraInput()

                    self.videoOutput.alwaysDiscardsLateVideoFrames = true
                    self.videoOutput.videoSettings = [
                        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                    ]
                    self.videoOutput.setSampleBufferDelegate(self.sampleBufferDelegate, queue: self.videoOutputQueue)

                    guard self.session.canAddOutput(self.videoOutput) else {
                        throw CameraError.cannotAddOutput
                    }
                    self.session.addOutput(self.videoOutput)

                    self.configureVideoConnection()

                    self.session.commitConfiguration()
                    self.didConfigureSession = true
                    continuation.resume()
                } catch {
                    self.session.commitConfiguration()
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func startRunning() {
        sessionQueue.async {
            guard !self.session.isRunning else { return }
            self.session.startRunning()
        }
    }

    func stopRunning() {
        sessionQueue.async {
            guard self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    private func loadModel() throws {
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            return
        }

        let model = try MLModel(contentsOf: modelURL)
        coreMLModel = model
        visionModel = try VNCoreMLModel(for: model)
        if let inputDescription = model.modelDescription.inputDescriptionsByName.values.first,
           let imageConstraint = inputDescription.imageConstraint {
            modelInputSize = CGSize(
                width: imageConstraint.pixelsWide,
                height: imageConstraint.pixelsHigh
            )
        }
    }

    private func processSampleBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        guard visionModel != nil else { return }
        guard !isProcessingFrame else { return }

        isProcessingFrame = true
        defer { isProcessingFrame = false }
        let handler = VNImageRequestHandler(
            cvPixelBuffer: imageBuffer,
            orientation: frameOrientation,
            options: [:]
        )
        let decodedResult = performPrediction(with: handler)
        let liveImageSize = detectionReferenceImageSize(for: imageBuffer)
        let trackedDetections = applyTracking(to: decodedResult.detections)
        let trails = makeTrackTrails(referenceTime: CACurrentMediaTime())
        let modelFPS = updateModelFPS()
        let resourceSnapshot = captureProcessResourceSnapshot()

        Task { @MainActor in
            self.delegate?.cameraCaptureController(
                self,
                didUpdateDetections: trackedDetections,
                trails: trails,
                imageSize: liveImageSize
            )
            self.delegate?.cameraCaptureController(
                self,
                didUpdateDebugInfo: DebugInfo(
                    modelName: self.modelName,
                    isModelLoaded: true,
                    confidenceThreshold: Double(self.currentConfidenceThreshold),
                    centerCropEnabled: self.currentCenterCropEnabled,
                    selectedCamera: self.currentSelectedCamera,
                    cameraFPS: self.currentCameraFPS,
                    maxMissedFramesForTracking: self.currentMaxMissedFramesForTracking,
                    trailDuration: self.currentTrailDurationOption,
                    imageSize: liveImageSize,
                    rawObservationCount: decodedResult.rawObservationCount,
                    filteredDetectionCount: trackedDetections.count,
                    rawPredictions: decodedResult.rawPredictions,
                    outputSummary: decodedResult.outputSummary,
                    rawRows: decodedResult.rawRows,
                    modelFPS: modelFPS,
                    activeTrackCount: activeTrackCount,
                    memoryUsageBytes: resourceSnapshot.memoryUsageBytes,
                    cpuUsagePercent: resourceSnapshot.cpuUsagePercent
                )
            )
        }
    }

    private func configureCameraInput() throws {
        let device = try resolveCameraDevice(for: selectedCamera)
        let cameraInput = try AVCaptureDeviceInput(device: device)

        guard session.canAddInput(cameraInput) else {
            throw CameraError.cannotAddInput
        }

        session.addInput(cameraInput)
        currentCameraInput = cameraInput
        try configureDevice(device)
    }

    private func replaceCameraInputIfNeeded() throws {
        let device = try resolveCameraDevice(for: selectedCamera)

        if currentCameraInput?.device.uniqueID == device.uniqueID {
            return
        }

        let newInput = try AVCaptureDeviceInput(device: device)

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        if let currentCameraInput {
            session.removeInput(currentCameraInput)
        }

        guard session.canAddInput(newInput) else {
            if let currentCameraInput, session.canAddInput(currentCameraInput) {
                session.addInput(currentCameraInput)
            }
            throw CameraError.cannotAddInput
        }

        session.addInput(newInput)
        configureVideoConnection()
        currentCameraInput = newInput
        try configureDevice(device)
        resetTracking()
    }

    private func configureVideoConnection() {
        guard let connection = videoOutput.connection(with: .video) else { return }
        connection.videoRotationAngle = 90
    }

    private func configureDevice(_ device: AVCaptureDevice) throws {
        try applyPreferredFrameRate(to: device)
    }

    private func resetTracking() {
        trackedObjects.removeAll()
        nextTrackID = 1
    }

    private func resolveCameraDevice(for option: CameraOption) throws -> AVCaptureDevice {
        let preferredPosition = option.position
        let deviceTypes: [AVCaptureDevice.DeviceType] = [
            .builtInWideAngleCamera,
            .builtInDualWideCamera,
            .builtInDualCamera,
            .builtInTripleCamera,
            .builtInUltraWideCamera,
            .builtInTrueDepthCamera
        ]
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes,
            mediaType: .video,
            position: preferredPosition
        )

        if let exactMatch = discovery.devices.first(where: { $0.position == preferredPosition && preferredPosition != .unspecified }) {
            return exactMatch
        }

        if let firstAvailable = discovery.devices.first {
            return firstAvailable
        }

        if preferredPosition == .unspecified,
           let fallback = AVCaptureDevice.default(for: .video) {
            return fallback
        }

        throw CameraError.noCameraAvailable(position: option)
    }

    private func applyPreferredFrameRate() throws {
        guard let device = currentCameraInput?.device else { return }
        try applyPreferredFrameRate(to: device)
    }

    private func applyPreferredFrameRate(to device: AVCaptureDevice) throws {
        let preferredFPS = Double(cameraFPS.value)
        let rankedFormat = supportedFormats(for: device, preferredFPS: preferredFPS)
            .max { lhs, rhs in
                if lhs.score == rhs.score {
                    return CMVideoFormatDescriptionGetDimensions(lhs.format.formatDescription).width <
                        CMVideoFormatDescriptionGetDimensions(rhs.format.formatDescription).width
                }
                return lhs.score < rhs.score
            }

        guard let rankedFormat else {
            throw CameraError.unsupportedFrameRate(cameraFPS, camera: selectedCamera)
        }

        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }

        device.activeFormat = rankedFormat.format
        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(cameraFPS.value))
        device.activeVideoMinFrameDuration = frameDuration
        device.activeVideoMaxFrameDuration = frameDuration
    }

    private func supportedFormats(
        for device: AVCaptureDevice,
        preferredFPS: Double
    ) -> [(format: AVCaptureDevice.Format, score: Int)] {
        device.formats.compactMap { format in
            let supportsFPS = format.videoSupportedFrameRateRanges.contains { range in
                range.minFrameRate <= preferredFPS && preferredFPS <= range.maxFrameRate
            }

            guard supportsFPS else { return nil }

            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let subtype = CMFormatDescriptionGetMediaSubType(format.formatDescription)
            let pixelFormatBonus = subtype == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
                subtype == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ? 10_000 : 0
            return (format, Int(dimensions.width) + pixelFormatBonus)
        }
    }
}

extension CameraViewModel: CameraCaptureControllerDelegate {
    func cameraCaptureController(
        _ controller: CameraCaptureController,
        didUpdateDetections detections: [Detection],
        trails: [TrackTrail],
        imageSize: CGSize
    ) {
        if imageSize != .zero {
            self.imageSize = imageSize
        }
        self.detections = detections
        self.trails = trails
    }

    func cameraCaptureController(
        _ controller: CameraCaptureController,
        didUpdateStatus statusText: String,
        errorMessage: String?
    ) {
        self.statusText = statusText
        self.errorMessage = errorMessage
    }

    func cameraCaptureController(
        _ controller: CameraCaptureController,
        didUpdateDebugInfo debugInfo: DebugInfo
    ) {
        var mergedInfo = debugInfo
        if imageSize != .zero {
            mergedInfo.imageSize = imageSize
        }
        self.debugInfo = mergedInfo
    }
}

private extension CameraCaptureController {
    var currentConfidenceThreshold: Float {
        confidenceThreshold
    }

    var currentCenterCropEnabled: Bool {
        centerCropEnabled
    }

    var currentSelectedCamera: CameraOption {
        selectedCamera
    }

    var currentCameraFPS: CameraFPSOption {
        cameraFPS
    }

    var currentMaxMissedFramesForTracking: Int {
        maxMissedFramesForTracking
    }

    var currentTrailDurationOption: TrailDurationOption {
        TrailDurationOption(duration: trailDuration)
    }

    func decodeYOLOOutput(
        _ multiArray: MLMultiArray,
        featureName: String,
        threshold: Float
    ) -> DecodedDetections {
        let shape = multiArray.shape.map(\.intValue)
        let reducedShape = shape.first == 1 ? Array(shape.dropFirst()) : shape

        if reducedShape == [300, 6] || reducedShape == [6, 300] {
            return decodeEndToEndDetections(
                multiArray,
                featureName: featureName,
                shape: shape,
                threshold: threshold
            )
        }

        guard reducedShape.count == 2 else {
            return DecodedDetections(
                detections: [],
                rawObservationCount: 0,
                rawPredictions: [],
                outputSummary: "\(featureName) shape \(shape)"
            )
        }

        let rows = reducedShape[0]
        let columns = reducedShape[1]
        let featuresFirst = rows <= columns
        let featureCount = featuresFirst ? rows : columns
        let candidateCount = featuresFirst ? columns : rows

        guard featureCount >= 5 else {
            return DecodedDetections(
                detections: [],
                rawObservationCount: 0,
                rawPredictions: [],
                outputSummary: "\(featureName) unsupported shape \(shape)"
            )
        }

        let labelNames = ["sphere"]
        var rawCandidates: [(label: String, confidence: Float, box: CGRect)] = []
        rawCandidates.reserveCapacity(candidateCount)
        var topRawScore: Float = -.greatestFiniteMagnitude
        var topNormalizedScore: Float = 0

        for candidateIndex in 0..<candidateCount {
            let x = value(in: multiArray, featuresFirst: featuresFirst, feature: 0, candidate: candidateIndex)
            let y = value(in: multiArray, featuresFirst: featuresFirst, feature: 1, candidate: candidateIndex)
            let width = value(in: multiArray, featuresFirst: featuresFirst, feature: 2, candidate: candidateIndex)
            let height = value(in: multiArray, featuresFirst: featuresFirst, feature: 3, candidate: candidateIndex)

            let classScores = (4..<featureCount).map {
                value(in: multiArray, featuresFirst: featuresFirst, feature: $0, candidate: candidateIndex)
            }

            guard let maxRawScore = classScores.max(), let classIndex = classScores.firstIndex(of: maxRawScore) else {
                continue
            }

            let normalizedScore = normalizeScore(maxRawScore)
            topRawScore = max(topRawScore, maxRawScore)
            topNormalizedScore = max(topNormalizedScore, normalizedScore)

            if normalizedScore <= 0.001 {
                continue
            }

            let label = classIndex < labelNames.count ? labelNames[classIndex] : "class \(classIndex)"
            let normalizedBox = normalizeBox(
                x: x,
                y: y,
                width: width,
                height: height
            )

            rawCandidates.append((label: label, confidence: normalizedScore, box: normalizedBox))
        }

        let sortedCandidates = rawCandidates.sorted { $0.confidence > $1.confidence }
        let detections = sortedCandidates
            .filter { $0.confidence >= threshold }
            .map {
                Detection(
                    id: UUID(),
                    label: $0.label,
                    confidence: $0.confidence,
                    boundingBox: $0.box,
                    trackID: nil
                )
            }

        let rawPredictions = sortedCandidates.prefix(3).map {
            "\($0.label) \(Int($0.confidence * 100))%"
        }

        return DecodedDetections(
            detections: applyNMS(to: detections),
            rawObservationCount: rawCandidates.count,
            rawPredictions: rawPredictions,
            outputSummary: "\(featureName) shape \(shape.map(String.init).joined(separator: "x")) maxRaw \(formatScore(topRawScore)) maxNorm \(formatScore(topNormalizedScore))"
        )
    }

    func decodeEndToEndDetections(
        _ multiArray: MLMultiArray,
        featureName: String,
        shape: [Int],
        threshold: Float
    ) -> DecodedDetections {
        let reducedShape = shape.first == 1 ? Array(shape.dropFirst()) : shape
        let detectionsFirst = reducedShape[0] == 300
        let detectionCount = detectionsFirst ? reducedShape[0] : reducedShape[1]
        let labelNames = ["sphere"]
        var rows: [[Float]] = []
        rows.reserveCapacity(detectionCount)
        var detections: [Detection] = []
        var topRawScore: Float = -.greatestFiniteMagnitude
        var topPredictions: [(String, Float)] = []

        for detectionIndex in 0..<detectionCount {
            let values = (0..<6).map { featureIndex in
                value(
                    in: multiArray,
                    featuresFirst: !detectionsFirst,
                    feature: featureIndex,
                    candidate: detectionIndex
                )
            }
            rows.append(values)

            let rawScore = values[4]
            topRawScore = max(topRawScore, rawScore)
            let score = normalizeScore(rawScore)
            if score <= 0.001 {
                continue
            }

            let rawClass = Int(round(values[5]))
            let label = rawClass >= 0 && rawClass < labelNames.count ? labelNames[rawClass] : "class \(rawClass)"
            topPredictions.append((label, score))

            guard score >= threshold else { continue }
            let box = normalizeCornerBox(
                x1: values[0],
                y1: values[1],
                x2: values[2],
                y2: values[3]
            )
            guard box.width > 0.001, box.height > 0.001 else { continue }

            detections.append(
                Detection(
                    id: UUID(),
                    label: label,
                    confidence: score,
                    boundingBox: box,
                    trackID: nil
                )
            )
        }

        let rawPredictions = topPredictions
            .sorted { $0.1 > $1.1 }
            .prefix(3)
            .map { "\($0.0) \(Int($0.1 * 100))%" }

        let columnMaxima = (0..<6).map { columnIndex in
            rows.map { $0[columnIndex] }.max() ?? 0
        }
        let columnMinima = (0..<6).map { columnIndex in
            rows.map { $0[columnIndex] }.min() ?? 0
        }

        return DecodedDetections(
            detections: applyNMS(to: detections),
            rawObservationCount: topPredictions.count,
            rawPredictions: rawPredictions,
            outputSummary: "\(featureName) shape \(shape.map(String.init).joined(separator: "x")) format xyxy conf c4 class c5 maxRaw \(formatScore(topRawScore))",
            rawRows: formattedRawRows(
                rows: rows,
                columnMinima: columnMinima,
                columnMaxima: columnMaxima
            )
        )
    }

    func value(
        in multiArray: MLMultiArray,
        featuresFirst: Bool,
        feature: Int,
        candidate: Int
    ) -> Float {
        let index: [NSNumber]
        switch multiArray.shape.count {
        case 3:
            index = featuresFirst
                ? [0, NSNumber(value: feature), NSNumber(value: candidate)]
                : [0, NSNumber(value: candidate), NSNumber(value: feature)]
        case 2:
            index = featuresFirst
                ? [NSNumber(value: feature), NSNumber(value: candidate)]
                : [NSNumber(value: candidate), NSNumber(value: feature)]
        default:
            let flatIndex = featuresFirst
                ? feature * max(candidate + 1, 1) + candidate
                : candidate * max(feature + 1, 1) + feature
            return multiArray[flatIndex].floatValue
        }
        return multiArray[index].floatValue
    }

    func normalizeBox(x: Float, y: Float, width: Float, height: Float) -> CGRect {
        let inputWidth = max(Float(modelInputSize.width), 1)
        let inputHeight = max(Float(modelInputSize.height), 1)

        let normalizedX = x > 1 ? x / inputWidth : x
        let normalizedY = y > 1 ? y / inputHeight : y
        let normalizedWidth = width > 1 ? width / inputWidth : width
        let normalizedHeight = height > 1 ? height / inputHeight : height

        let minX = max(0, normalizedX - normalizedWidth / 2)
        let minY = max(0, normalizedY - normalizedHeight / 2)
        let boxWidth = min(1 - minX, normalizedWidth)
        let boxHeight = min(1 - minY, normalizedHeight)

        return CGRect(
            x: CGFloat(minX),
            y: CGFloat(minY),
            width: CGFloat(max(0, boxWidth)),
            height: CGFloat(max(0, boxHeight))
        )
    }

    func normalizeCornerBox(x1: Float, y1: Float, x2: Float, y2: Float) -> CGRect {
        let inputWidth = max(Float(modelInputSize.width), 1)
        let inputHeight = max(Float(modelInputSize.height), 1)

        let normalizedX1 = x1 > 1 ? x1 / inputWidth : x1
        let normalizedY1 = y1 > 1 ? y1 / inputHeight : y1
        let normalizedX2 = x2 > 1 ? x2 / inputWidth : x2
        let normalizedY2 = y2 > 1 ? y2 / inputHeight : y2

        let minX = max(0, min(normalizedX1, normalizedX2))
        let minY = max(0, min(normalizedY1, normalizedY2))
        let maxX = min(1, max(normalizedX1, normalizedX2))
        let maxY = min(1, max(normalizedY1, normalizedY2))

        return CGRect(
            x: CGFloat(minX),
            y: CGFloat(minY),
            width: CGFloat(max(0, maxX - minX)),
            height: CGFloat(max(0, maxY - minY))
        )
    }

    func normalizeScore(_ rawScore: Float) -> Float {
        if (0...1).contains(rawScore) {
            return rawScore
        }
        return 1 / (1 + exp(-rawScore))
    }

    func performPrediction(with handler: VNImageRequestHandler) -> DecodedDetections {
        guard let visionModel else {
            return DecodedDetections(
                detections: [],
                rawObservationCount: 0,
                rawPredictions: [],
                outputSummary: "Kein Vision-Modell"
            )
        }

        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = currentCenterCropEnabled ? .centerCrop : .scaleFill

        do {
            try handler.perform([request])
        } catch {
            return DecodedDetections(
                detections: [],
                rawObservationCount: 0,
                rawPredictions: [],
                outputSummary: "Vision-Fehler: \(error.localizedDescription)"
            )
        }

        let featureObservations = request.results?
            .compactMap { $0 as? VNCoreMLFeatureValueObservation } ?? []

        if let firstMultiArrayObservation = featureObservations.first(where: { $0.featureValue.multiArrayValue != nil }),
           let multiArray = firstMultiArrayObservation.featureValue.multiArrayValue {
            return decodeYOLOOutput(
                multiArray,
                featureName: firstMultiArrayObservation.featureName,
                threshold: currentConfidenceThreshold
            )
        }

        let featureNames = featureObservations.map(\.featureName)
        return DecodedDetections(
            detections: [],
            rawObservationCount: 0,
            rawPredictions: [],
            outputSummary: "Prediction ohne MultiArray",
            rawRows: ["featureNames: \(featureNames)"]
        )
    }

    func detectionReferenceImageSize(for imageBuffer: CVPixelBuffer) -> CGSize {
        let width = CGFloat(CVPixelBufferGetWidth(imageBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(imageBuffer))

        if currentCenterCropEnabled {
            let sideLength = min(width, height)
            return CGSize(width: sideLength, height: sideLength)
        }

        return CGSize(width: modelInputSize.width, height: modelInputSize.height)
    }

    func captureProcessResourceSnapshot() -> ProcessResourceSnapshot {
        ProcessResourceSnapshot(
            memoryUsageBytes: currentMemoryUsageBytes(),
            cpuUsagePercent: currentCPUUsagePercent()
        )
    }

    func currentMemoryUsageBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<integer_t>.stride
        )

        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    reboundPointer,
                    &count
                )
            }
        }

        guard result == KERN_SUCCESS else { return 0 }
        return info.phys_footprint
    }

    func currentCPUUsagePercent() -> Double {
        var threads: thread_act_array_t?
        var threadCount: mach_msg_type_number_t = 0

        let result = task_threads(mach_task_self_, &threads, &threadCount)
        guard result == KERN_SUCCESS, let threads else { return 0 }

        defer {
            vm_deallocate(
                mach_task_self_,
                vm_address_t(bitPattern: threads),
                vm_size_t(Int(threadCount) * MemoryLayout<thread_t>.stride)
            )
        }

        var totalCPUUsagePercent = 0.0

        for index in 0..<Int(threadCount) {
            var threadInfo = thread_basic_info_data_t()
            var threadInfoCount = mach_msg_type_number_t(THREAD_INFO_MAX)

            let threadInfoResult = withUnsafeMutablePointer(to: &threadInfo) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(threadInfoCount)) { reboundPointer in
                    thread_info(
                        threads[index],
                        thread_flavor_t(THREAD_BASIC_INFO),
                        reboundPointer,
                        &threadInfoCount
                    )
                }
            }

            guard threadInfoResult == KERN_SUCCESS else { continue }
            guard (threadInfo.flags & TH_FLAGS_IDLE) == 0 else { continue }

            totalCPUUsagePercent += Double(threadInfo.cpu_usage) / Double(TH_USAGE_SCALE) * 100
        }

        return totalCPUUsagePercent
    }

    func formatScore(_ value: Float) -> String {
        guard value.isFinite else { return "-" }
        return String(format: "%.3f", value)
    }

    func updateModelFPS() -> Double {
        let now = CACurrentMediaTime()
        defer { lastInferenceTimestamp = now }

        guard let lastInferenceTimestamp else {
            return smoothedFPS
        }

        let delta = now - lastInferenceTimestamp
        guard delta > 0 else {
            return smoothedFPS
        }

        let instantFPS = 1.0 / delta
        smoothedFPS = smoothedFPS == 0 ? instantFPS : (smoothedFPS * 0.8) + (instantFPS * 0.2)
        return smoothedFPS
    }

    func formattedRawRows(
        rows: [[Float]],
        columnMinima: [Float],
        columnMaxima: [Float]
    ) -> [String] {
        var output = (0..<6).map { index in
            "c\(index) min \(formatScore(columnMinima[index])) max \(formatScore(columnMaxima[index]))"
        }

        let sampleRows = rows.prefix(3).enumerated().map { offset, row in
            let values = row.map(formatScore).joined(separator: ", ")
            return "r\(offset): [\(values)]"
        }
        output.append(contentsOf: sampleRows)
        return output
    }

    func applyNMS(to detections: [Detection], iouThreshold: CGFloat = 0.45) -> [Detection] {
        var kept: [Detection] = []

        for detection in detections.sorted(by: { $0.confidence > $1.confidence }) {
            let overlaps = kept.contains {
                $0.label == detection.label &&
                $0.boundingBox.intersection(detection.boundingBox).area / max($0.boundingBox.union(detection.boundingBox).area, 0.0001) > iouThreshold
            }

            if !overlaps {
                kept.append(detection)
            }
        }

        return kept
    }

    var activeTrackCount: Int {
        trackedObjects.count
    }

    func applyTracking(to detections: [Detection]) -> [Detection] {
        let matchThreshold: CGFloat = 0.3
        let now = CACurrentMediaTime()

        for index in trackedObjects.indices {
            trackedObjects[index].missedFrames += 1
        }

        var availableTrackIndices = Set(trackedObjects.indices)
        let sortedDetections = detections.sorted { $0.confidence > $1.confidence }
        var trackedDetections: [Detection] = []
        trackedDetections.reserveCapacity(sortedDetections.count)

        for detection in sortedDetections {
            let bestMatch = availableTrackIndices
                .compactMap { trackIndex -> (Int, CGFloat)? in
                    let track = trackedObjects[trackIndex]
                    guard track.label == detection.label else { return nil }
                    let iou = intersectionOverUnion(track.boundingBox, detection.boundingBox)
                    guard iou >= matchThreshold else { return nil }
                    return (trackIndex, iou)
                }
                .max(by: { $0.1 < $1.1 })

            if let (trackIndex, _) = bestMatch {
                availableTrackIndices.remove(trackIndex)
                trackedObjects[trackIndex].boundingBox = detection.boundingBox
                trackedObjects[trackIndex].confidence = detection.confidence
                trackedObjects[trackIndex].missedFrames = 0
                trackedObjects[trackIndex].history.append(
                    TimedPoint(
                        timestamp: now,
                        point: detection.boundingBox.center
                    )
                )
                trackedDetections.append(
                    Detection(
                        id: detection.id,
                        label: detection.label,
                        confidence: detection.confidence,
                        boundingBox: detection.boundingBox,
                        trackID: trackedObjects[trackIndex].trackID
                    )
                )
                continue
            }

            let newTrack = TrackedObject(
                trackID: nextTrackID,
                label: detection.label,
                confidence: detection.confidence,
                boundingBox: detection.boundingBox,
                history: [
                    TimedPoint(
                        timestamp: now,
                        point: detection.boundingBox.center
                    )
                ]
            )
            nextTrackID += 1
            trackedObjects.append(newTrack)
            trackedDetections.append(
                Detection(
                    id: detection.id,
                    label: detection.label,
                    confidence: detection.confidence,
                    boundingBox: detection.boundingBox,
                    trackID: newTrack.trackID
                )
            )
        }

        trackedObjects.removeAll { $0.missedFrames > maxMissedFramesForTracking }
        trimTrackHistories(referenceTime: now)
        return trackedDetections
    }

    func trimTrackHistories(referenceTime: TimeInterval) {
        guard trailDuration > 0 else {
            for index in trackedObjects.indices {
                trackedObjects[index].history.removeAll()
            }
            return
        }

        let cutoff = referenceTime - trailDuration
        for index in trackedObjects.indices {
            trackedObjects[index].history.removeAll { $0.timestamp < cutoff }
        }
    }

    func makeTrackTrails(referenceTime: TimeInterval) -> [TrackTrail] {
        trimTrackHistories(referenceTime: referenceTime)
        guard trailDuration > 0 else { return [] }

        return trackedObjects.compactMap { trackedObject in
            let points = trackedObject.history.map(\.point)
            guard points.count >= 2 else { return nil }
            return TrackTrail(trackID: trackedObject.trackID, points: points)
        }
    }

    func intersectionOverUnion(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
        let intersectionArea = lhs.intersection(rhs).area
        guard intersectionArea > 0 else { return 0 }
        let unionArea = lhs.union(rhs).area
        guard unionArea > 0 else { return 0 }
        return intersectionArea / unionArea
    }
}

struct Detection: Identifiable, Sendable {
    let id: UUID
    let label: String
    let confidence: VNConfidence
    let boundingBox: CGRect
    let trackID: Int?
}

struct DebugInfo: Sendable {
    let modelName: String
    let isModelLoaded: Bool
    var confidenceThreshold: Double = 0.35
    var centerCropEnabled = false
    var selectedCamera: CameraOption = .back
    var cameraFPS: CameraFPSOption = .fps30
    var maxMissedFramesForTracking = defaultTrackToleranceValue
    var trailDuration: TrailDurationOption = .off
    var imageSize: CGSize = .zero
    var rawObservationCount = 0
    var filteredDetectionCount = 0
    var rawPredictions: [String] = []
    var outputSummary = "-"
    var rawRows: [String] = []
    var modelFPS: Double = 0
    var activeTrackCount = 0
    var memoryUsageBytes: UInt64 = 0
    var cpuUsagePercent: Double = 0

    var imageSizeText: String {
        guard imageSize != .zero else { return "-" }
        return "\(Int(imageSize.width)) x \(Int(imageSize.height))"
    }

    var cameraTitle: String {
        selectedCamera.title
    }

    var cameraFPSText: String {
        cameraFPS.title
    }

    var topPredictionText: String {
        rawPredictions.first ?? "-"
    }

    var modelFPSText: String {
        String(format: "%.1f", modelFPS)
    }

    var memoryUsageText: String {
        guard memoryUsageBytes > 0 else { return "-" }
        let megabytes = Double(memoryUsageBytes) / 1_048_576
        return String(format: "%.1f MB", megabytes)
    }

    var cpuUsageText: String {
        String(format: "%.1f %%", cpuUsagePercent)
    }
}

private struct DecodedDetections {
    let detections: [Detection]
    let rawObservationCount: Int
    let rawPredictions: [String]
    let outputSummary: String
    var rawRows: [String] = []
}

private struct ProcessResourceSnapshot: Sendable {
    let memoryUsageBytes: UInt64
    let cpuUsagePercent: Double
}

private struct TrackedObject: Sendable {
    let trackID: Int
    let label: String
    var confidence: VNConfidence
    var boundingBox: CGRect
    var missedFrames = 0
    var history: [TimedPoint] = []
}

private struct TimedPoint: Sendable {
    let timestamp: TimeInterval
    let point: CGPoint
}

struct TrackTrail: Identifiable, Sendable {
    let trackID: Int
    let points: [CGPoint]

    var id: Int { trackID }
}

private extension CGRect {
    var area: CGFloat {
        width * height
    }
}

private final class VideoOutputDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let onSampleBuffer: @Sendable (CMSampleBuffer) -> Void

    init(onSampleBuffer: @escaping @Sendable (CMSampleBuffer) -> Void) {
        self.onSampleBuffer = onSampleBuffer
    }

    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        onSampleBuffer(sampleBuffer)
    }
}

private enum CameraError: LocalizedError {
    case noCameraAvailable(position: CameraOption)
    case cannotAddInput
    case cannotAddOutput
    case unsupportedFrameRate(CameraFPSOption, camera: CameraOption)

    var errorDescription: String? {
        switch self {
        case .noCameraAvailable(let position):
            return "Keine Kamera fuer '\(position.title)' verfuegbar."
        case .cannotAddInput:
            return "Kamera-Input konnte der Session nicht hinzugefuegt werden."
        case .cannotAddOutput:
            return "Video-Output konnte der Session nicht hinzugefuegt werden."
        case .unsupportedFrameRate(let fps, let camera):
            return "\(fps.title) wird von '\(camera.title)' nicht unterstuetzt."
        }
    }
}

enum CameraFPSOption: Int, CaseIterable, Identifiable, Sendable {
    case fps1 = 1
    case fps2 = 2
    case fps5 = 5
    case fps10 = 10
    case fps15 = 15
    case fps24 = 24
    case fps25 = 25
    case fps30 = 30
    case fps48 = 48
    case fps50 = 50
    case fps60 = 60
    case fps90 = 90
    case fps120 = 120
    case fps240 = 240

    var id: Int { rawValue }

    var value: Int { rawValue }

    var title: String {
        "\(rawValue) FPS"
    }
}

let supportedTrackToleranceValues = [0, 1, 2, 5, 10, 15, 25, 30, 60, 120, 240]
let defaultTrackToleranceValue = 5

func snappedTrackToleranceValue(_ value: Int) -> Int {
    supportedTrackToleranceValues.min { lhs, rhs in
        let lhsDistance = abs(lhs - value)
        let rhsDistance = abs(rhs - value)
        if lhsDistance == rhsDistance {
            return lhs < rhs
        }
        return lhsDistance < rhsDistance
    } ?? defaultTrackToleranceValue
}

enum CameraOption: String, CaseIterable, Identifiable, Sendable {
    case back
    case front
    case automatic

    var id: String { rawValue }

    var title: String {
        switch self {
        case .back:
            return "Hinten"
        case .front:
            return "Vorne"
        case .automatic:
            return "Auto"
        }
    }

    var position: AVCaptureDevice.Position {
        switch self {
        case .back:
            return .back
        case .front:
            return .front
        case .automatic:
            return .unspecified
        }
    }
}

enum TrailDurationOption: String, CaseIterable, Identifiable, Sendable {
    case off
    case halfSecond
    case oneSecond
    case oneAndHalfSeconds
    case twoSeconds
    case threeSeconds
    case fiveSeconds

    var id: String { rawValue }

    var duration: TimeInterval {
        switch self {
        case .off:
            return 0
        case .halfSecond:
            return 0.5
        case .oneSecond:
            return 1
        case .oneAndHalfSeconds:
            return 1.5
        case .twoSeconds:
            return 2
        case .threeSeconds:
            return 3
        case .fiveSeconds:
            return 5
        }
    }

    var title: String {
        switch self {
        case .off:
            return "0 s"
        case .halfSecond:
            return "0,5 s"
        case .oneSecond:
            return "1 s"
        case .oneAndHalfSeconds:
            return "1,5 s"
        case .twoSeconds:
            return "2 s"
        case .threeSeconds:
            return "3 s"
        case .fiveSeconds:
            return "5 s"
        }
    }

    init(duration: TimeInterval) {
        switch duration {
        case 0.5:
            self = .halfSecond
        case 1:
            self = .oneSecond
        case 1.5:
            self = .oneAndHalfSeconds
        case 2:
            self = .twoSeconds
        case 3:
            self = .threeSeconds
        case 5:
            self = .fiveSeconds
        default:
            self = .off
        }
    }
}

private struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let videoGravity: AVLayerVideoGravity

    func makeUIView(context: Context) -> PreviewContainerView {
        let view = PreviewContainerView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = videoGravity
        view.updateOrientation()
        return view
    }

    func updateUIView(_ uiView: PreviewContainerView, context: Context) {
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
        uiView.previewLayer.videoGravity = videoGravity
        uiView.updateOrientation()
    }
}

private final class PreviewContainerView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    func updateOrientation() {
        guard let connection = previewLayer.connection else { return }
        if connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
    }
}

private struct DetectionOverlay: View {
    let detections: [Detection]
    let trails: [TrackTrail]
    let imageSize: CGSize

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                ForEach(trails) { trail in
                    trailPath(for: trail, in: geometry.size)
                }

                ForEach(Array(detections.enumerated()), id: \.element.id) { _, detection in
                    overlayBox(for: detection, in: geometry.size)
                }
            }
        }
    }

    @ViewBuilder
    private func overlayBox(for detection: Detection, in viewSize: CGSize) -> some View {
        let rect = rectForDetection(detection.boundingBox, in: viewSize)

        ZStack(alignment: .topTrailing) {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(.green, lineWidth: 3)
                .frame(width: rect.width, height: rect.height)

            Text(overlayTitle(for: detection))
                .font(.caption2.monospacedDigit().weight(.semibold))
                .foregroundStyle(.black)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.green)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .padding(6)
        }
        .frame(width: rect.width, height: rect.height)
        .position(x: rect.midX, y: rect.midY)
    }

    private func rectForDetection(_ boundingBox: CGRect, in viewSize: CGSize) -> CGRect {
        guard imageSize != .zero else { return .zero }

        return CGRect(
            x: boundingBox.minX * viewSize.width,
            y: boundingBox.minY * viewSize.height,
            width: boundingBox.width * viewSize.width,
            height: boundingBox.height * viewSize.height
        )
    }

    @ViewBuilder
    private func trailPath(for trail: TrackTrail, in viewSize: CGSize) -> some View {
        let points = trail.points.map { point in
            CGPoint(x: point.x * viewSize.width, y: point.y * viewSize.height)
        }

        Path { path in
            guard let firstPoint = points.first else { return }
            path.move(to: firstPoint)
            for point in points.dropFirst() {
                path.addLine(to: point)
            }
        }
        .stroke(.green.opacity(0.9), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
    }

    private func overlayTitle(for detection: Detection) -> String {
        let trackPrefix = detection.trackID.map { "#\($0) " } ?? ""
        return "\(trackPrefix)\(detection.label) \(Int((detection.confidence * 100).rounded()))%"
    }
}

private extension CGRect {
    var center: CGPoint {
        CGPoint(x: midX, y: midY)
    }
}

#Preview {
    ContentView()
}
