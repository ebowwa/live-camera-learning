import Foundation

struct AppStoreApp: Codable, Identifiable {
    let id: String
    let type: String
    let attributes: AppAttributes
    
    struct AppAttributes: Codable {
        let name: String
        let bundleId: String
        let sku: String?
        let primaryLocale: String?
        let isOrEverWasMadeForKids: Bool?
        let subscriptionStatusUrl: String?
        let subscriptionStatusUrlVersion: String?
        let subscriptionStatusUrlForSandbox: String?
        let subscriptionStatusUrlVersionForSandbox: String?
        let contentRightsDeclaration: String?
        let streamlinedPurchasingEnabled: Bool?
        let appStoreVersions: AppStoreVersions?
    }
    
    struct AppStoreVersions: Codable {
        let links: Links?
        
        struct Links: Codable {
            let `self`: String?
            let related: String?
        }
    }
}

struct AppsResponse: Codable {
    let data: [AppStoreApp]
    let links: ResponseLinks?
    let meta: ResponseMeta?
    
    struct ResponseLinks: Codable {
        let `self`: String?
        let first: String?
        let next: String?
    }
    
    struct ResponseMeta: Codable {
        let paging: Paging?
        
        struct Paging: Codable {
            let total: Int?
            let limit: Int?
        }
    }
}

struct AppStoreVersion: Codable, Identifiable {
    let id: String
    let type: String
    let attributes: VersionAttributes
    
    struct VersionAttributes: Codable {
        let platform: String?
        let versionString: String?
        let appStoreState: String?
        let storeIcon: StoreIcon?
        let watchStoreIcon: StoreIcon?
        let copyright: String?
        let releaseType: String?
        let earliestReleaseDate: String?
        let usesIdfa: Bool?
        let downloadable: Bool?
        let createdDate: String?
        
        struct StoreIcon: Codable {
            let templateUrl: String?
            let width: Int?
            let height: Int?
        }
    }
}

struct AppInfo: Codable, Identifiable {
    let id: String
    let type: String
    let attributes: InfoAttributes
    
    struct InfoAttributes: Codable {
        let appStoreState: String?
        let appStoreAgeRating: String?
        let brazilAgeRating: String?
        let kidsAgeBand: String?
    }
}
