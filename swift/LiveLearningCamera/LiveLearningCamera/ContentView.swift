//
//  ContentView.swift
//  LiveLearningCamera
//
//  Created by Elijah Arbee on 8/11/25.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var apiClient = AppStoreConnectAPI()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                VStack(spacing: 10) {
                    Text("App Store Connect API Test")
                        .font(.title)
                        .fontWeight(.bold)
                    
                    Text("LiveLearningCamera")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    Text("Bundle ID: ebowwa.LiveLearningCamera")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                
                VStack(spacing: 15) {
                    HStack {
                        Text("Connection Status:")
                            .fontWeight(.medium)
                        Spacer()
                        Text(apiClient.connectionStatus)
                            .foregroundColor(apiClient.connectionStatus.contains("✅") ? .green : 
                                           apiClient.connectionStatus.contains("❌") ? .red : .orange)
                    }
                    .padding(.horizontal)
                    
                    if apiClient.isLoading {
                        ProgressView("Testing API connection...")
                            .padding()
                    } else if let errorMessage = apiClient.errorMessage {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Error Details:")
                                .fontWeight(.medium)
                                .foregroundColor(.red)
                            Text(errorMessage)
                                .font(.caption)
                                .foregroundColor(.red)
                                .padding(.horizontal)
                        }
                        .padding()
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(8)
                    }
                    
                    if !apiClient.apps.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Apps Found: \(apiClient.apps.count)")
                                .fontWeight(.medium)
                                .padding(.horizontal)
                            
                            List(apiClient.apps) { app in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(app.attributes.name)
                                        .font(.headline)
                                    Text(app.attributes.bundleId)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    if let sku = app.attributes.sku {
                                        Text("SKU: \(sku)")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                    }
                                }
                                .padding(.vertical, 2)
                            }
                            .frame(maxHeight: 300)
                        }
                    }
                }
                
                Spacer()
                
                VStack(spacing: 12) {
                    Button("Test API Connection") {
                        Task {
                            await apiClient.testConnection()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(apiClient.isLoading)
                    
                    Button("Fetch All Apps") {
                        Task {
                            await apiClient.fetchApps()
                        }
                    }
                    .buttonStyle(.bordered)
                    .disabled(apiClient.isLoading)
                }
                .padding()
            }
            .navigationTitle("API Test")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task {
            await apiClient.testConnection()
        }
    }
}

#Preview {
    ContentView()
}
