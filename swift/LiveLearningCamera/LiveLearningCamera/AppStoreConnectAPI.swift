import Foundation
import CryptoKit
import Combine

@MainActor
class AppStoreConnectAPI: ObservableObject {
    @Published var apps: [AppStoreApp] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var connectionStatus: String = "Not connected"
    
    private let baseURL = "https://api.appstoreconnect.apple.com"
    private let keyID: String
    private let issuerID: String = "69a6de92-4f8b-47e3-e053-5b8c7c11a4d1"
    private let privateKey: String
    
    init() {
        self.keyID = ProcessInfo.processInfo.environment["APPSTORE_CONNECT_KEY_ID"] ?? ""
        self.privateKey = ProcessInfo.processInfo.environment["APPSTORE_CONNECT_PRIVATE_KEY"] ?? ""
        
        if keyID.isEmpty || privateKey.isEmpty {
            self.errorMessage = "Missing App Store Connect credentials"
            self.connectionStatus = "Missing credentials"
        }
    }
    
    private func generateJWT() throws -> String {
        guard !keyID.isEmpty, !privateKey.isEmpty else {
            throw APIError.missingCredentials
        }
        
        let header = JWTHeader(alg: "ES256", kid: keyID, typ: "JWT")
        let payload = JWTPayload(
            iss: issuerID,
            exp: Int(Date().timeIntervalSince1970) + 1200,
            aud: "appstoreconnect-v1"
        )
        
        let headerData = try JSONEncoder().encode(header)
        let payloadData = try JSONEncoder().encode(payload)
        
        let headerString = headerData.base64URLEncodedString()
        let payloadString = payloadData.base64URLEncodedString()
        let signingInput = "\(headerString).\(payloadString)"
        
        let cleanedPrivateKey = privateKey
            .replacingOccurrences(of: "\\n", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        
        let key = try P256.Signing.PrivateKey(pemRepresentation: cleanedPrivateKey)
        let signatureData = try key.signature(for: Data(signingInput.utf8))
        
        let derSignature = signatureData.derRepresentation
        let signatureString = derSignature.base64URLEncodedString()
        
        return "\(signingInput).\(signatureString)"
    }
    
    func testConnection() async {
        isLoading = true
        errorMessage = nil
        connectionStatus = "Testing connection..."
        
        do {
            let jwt = try generateJWT()
            
            guard let url = URL(string: "\(baseURL)/v1/apps?limit=5") else {
                throw APIError.invalidURL
            }
            
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }
            
            if httpResponse.statusCode == 200 {
                connectionStatus = "✅ Connected successfully"
                let appsResponse = try JSONDecoder().decode(AppsResponse.self, from: data)
                self.apps = appsResponse.data
            } else {
                let errorData = String(data: data, encoding: .utf8) ?? "Unknown error"
                throw APIError.httpError(httpResponse.statusCode, errorData)
            }
            
        } catch {
            connectionStatus = "❌ Connection failed"
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    func fetchApps() async {
        isLoading = true
        errorMessage = nil
        
        do {
            let jwt = try generateJWT()
            
            guard let url = URL(string: "\(baseURL)/v1/apps") else {
                throw APIError.invalidURL
            }
            
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }
            
            if httpResponse.statusCode == 200 {
                let appsResponse = try JSONDecoder().decode(AppsResponse.self, from: data)
                self.apps = appsResponse.data
            } else {
                let errorData = String(data: data, encoding: .utf8) ?? "Unknown error"
                throw APIError.httpError(httpResponse.statusCode, errorData)
            }
            
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
}

struct JWTHeader: Codable {
    let alg: String
    let kid: String
    let typ: String
}

struct JWTPayload: Codable {
    let iss: String
    let exp: Int
    let aud: String
}

enum APIError: LocalizedError {
    case invalidURL
    case invalidResponse
    case httpError(Int, String)
    case missingCredentials
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .invalidResponse:
            return "Invalid response"
        case .httpError(let code, let message):
            return "HTTP Error \(code): \(message)"
        case .missingCredentials:
            return "Missing App Store Connect credentials"
        }
    }
}

extension Data {
    func base64URLEncodedString() -> String {
        return self.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}
