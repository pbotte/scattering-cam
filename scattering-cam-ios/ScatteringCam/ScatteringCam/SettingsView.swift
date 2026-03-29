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

                    Stepper(value: $maxMissedFramesForTracking, in: 0...20) {
                        LabeledContent("Track-Toleranz") {
                            Text("\(maxMissedFramesForTracking) Frames")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
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
