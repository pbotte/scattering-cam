//
//  SettingsView.swift
//  ScatteringCam
//
//  Created by OpenAI Codex on 29.03.26.
//

import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss

    @Binding var confidenceThreshold: Double
    @Binding var maxMissedFramesForTracking: Int
    @Binding var trailDuration: TrailDurationOption
    @Binding var centerCropEnabled: Bool
    @Binding var selectedCamera: CameraOption
    @Binding var cameraFPS: CameraFPSOption

    var body: some View {
        NavigationStack {
            Form {
                Section("Kamera") {
                    Picker("Kamera", selection: $selectedCamera) {
                        ForEach(CameraOption.allCases) { option in
                            Text(option.title).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)

                    Toggle("640x640 Ausschnitt", isOn: $centerCropEnabled)

                    Picker("FPS", selection: $cameraFPS) {
                        ForEach(CameraFPSOption.allCases) { option in
                            Text(option.title).tag(option)
                        }
                    }
                }

                Section("Erkennung") {
                    VStack(alignment: .leading, spacing: 8) {
                        LabeledContent("Schwellwert") {
                            Text("\(Int(confidenceThreshold * 100))%")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }

                        Slider(
                            value: $confidenceThreshold,
                            in: 0...1,
                            step: 0.05
                        )
                    }

                    Picker("Track-Toleranz", selection: $maxMissedFramesForTracking) {
                        ForEach(supportedTrackToleranceValues, id: \.self) { value in
                            Text("\(value) Frames").tag(value)
                        }
                    }

                    Picker("Spur", selection: $trailDuration) {
                        ForEach(TrailDurationOption.allCases) { option in
                            Text(option.title).tag(option)
                        }
                    }
                }
            }
            .navigationTitle("Einstellungen")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Fertig") {
                        dismiss()
                    }
                }
            }
        }
    }
}
