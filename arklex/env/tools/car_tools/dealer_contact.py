"""
Dealer Contact and Locator Tool
Comprehensive tool for finding car dealers with contact information, location data, and services
"""

import json
import logging
import sqlite3
import math
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from arklex.env.tools.tools import register_tool

logger = logging.getLogger(__name__)

def _get_database_connection() -> sqlite3.Connection:
    """Get connection to the car database"""
    current_path = Path(__file__).resolve()
    project_root = current_path
    while project_root.parent != project_root:
        if (project_root / "pyproject.toml").exists() or (project_root / "requirements.txt").exists():
            break
        project_root = project_root.parent
    
    db_path = project_root / "examples" / "car_advisor" / "car_advisor_db.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Car database not found at {db_path}. Please run build_car_database.py first.")
    
    return sqlite3.connect(str(db_path))

def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula (in miles)"""
    if not all([lat1, lon1, lat2, lon2]):
        return float('inf')
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Earth's radius in miles
    earth_radius_miles = 3959
    distance = earth_radius_miles * c
    
    return round(distance, 1)

def _get_coordinates_for_location(location: str) -> Optional[Tuple[float, float]]:
    """Get approximate coordinates for common US locations (comprehensive lookup)"""
    location_coords = {
        # Major cities
        "new york": (40.7128, -74.0060),
        "new york city": (40.7128, -74.0060),
        "nyc": (40.7128, -74.0060),
        "los angeles": (34.0522, -118.2437),
        "la": (34.0522, -118.2437),
        "chicago": (41.8781, -87.6298),
        "houston": (29.7604, -95.3698),
        "phoenix": (33.4484, -112.0740),
        "philadelphia": (39.9526, -75.1652),
        "san antonio": (29.4241, -98.4936),
        "san diego": (32.7157, -117.1611),
        "dallas": (32.7767, -96.7970),
        "san jose": (37.3382, -121.8863),
        "austin": (30.2672, -97.7431),
        "jacksonville": (30.3322, -81.6557),
        "san francisco": (37.7749, -122.4194),
        "columbus": (39.9612, -82.9988),
        "fort worth": (32.7555, -97.3308),
        "charlotte": (35.2271, -80.8431),
        "seattle": (47.6062, -122.3321),
        "denver": (39.7392, -104.9903),
        "boston": (42.3601, -71.0589),
        "detroit": (42.3314, -83.0458),
        "nashville": (36.1627, -86.7816),
        "memphis": (35.1495, -90.0490),
        "las vegas": (36.1699, -115.1398),
        "miami": (25.7617, -80.1918),
        "atlanta": (33.7490, -84.3880),
        
        # States (approximate centers)
        "california": (36.7783, -119.4179),
        "texas": (31.9686, -99.9018),
        "florida": (27.7663, -82.6404),
        "new york": (40.7128, -74.0060),
        "pennsylvania": (41.2033, -77.1945),
        "illinois": (40.6331, -89.3985),
        "ohio": (40.4173, -82.9071),
        "georgia": (32.1656, -82.9001),
        "north carolina": (35.7596, -79.0193),
        "michigan": (44.3467, -85.4102),
        "new jersey": (40.0583, -74.4057),
        "virginia": (37.4316, -78.6569),
        "washington": (47.7511, -120.7401),
        "arizona": (34.0489, -111.0937),
        "massachusetts": (42.4072, -71.3824),
        "tennessee": (35.5175, -86.5804),
        "indiana": (40.2731, -86.1349),
        "missouri": (37.9643, -91.8318),
        "maryland": (39.0458, -76.6413),
        "wisconsin": (43.7844, -88.7879),
        "colorado": (39.5501, -105.7821),
        "minnesota": (46.7296, -94.6859),
        "south carolina": (33.8361, -81.1637),
        "alabama": (32.3182, -86.9023),
        "louisiana": (30.9843, -91.9623),
        "kentucky": (37.8393, -84.2700),
        "oregon": (43.8041, -120.5542),
        "oklahoma": (35.0078, -97.0929),
        "connecticut": (41.6032, -73.0877),
        "utah": (39.3210, -111.0937),
        "iowa": (41.8780, -93.0977),
        "nevada": (38.8026, -116.4194),
        "arkansas": (35.2010, -91.8318),
        "mississippi": (32.3547, -89.3985),
        "kansas": (39.0119, -98.4842),
        "new mexico": (34.5199, -105.8701),
        "nebraska": (41.4925, -99.9018),
        "west virginia": (38.5976, -80.4549),
        "idaho": (44.0682, -114.7420),
        "hawaii": (19.8968, -155.5828),
        "new hampshire": (43.1939, -71.5724),
        "maine": (45.2538, -69.4455),
        "montana": (47.0527, -110.2145),
        "rhode island": (41.5801, -71.4774),
        "delaware": (38.9108, -75.5277),
        "south dakota": (43.9695, -99.9018),
        "north dakota": (47.5515, -101.0020),
        "alaska": (64.0685, -152.2782),
        "vermont": (44.2601, -72.5806),
        "wyoming": (43.0759, -107.2903)
    }
    
    location_lower = location.lower().strip()
    
    # Direct lookup
    if location_lower in location_coords:
        return location_coords[location_lower]
    
    # Try partial matches for cities
    for key, coords in location_coords.items():
        if location_lower in key or key in location_lower:
            return coords
    
    # If no match found, return None (user will need to provide coordinates)
    return None

@register_tool(
    "Comprehensive dealer finder with contact information, location data, distances, services, ratings, and price negotiation analysis",
    [
        {
            "name": "location",
            "type": "str",
            "description": "User location (city, state, or zip code)",
            "prompt": "What is your location (city, state, or zip code)?",
            "required": True,
        },
        {
            "name": "search_type",
            "type": "str",
            "description": "Type of search: 'contact_info', 'nearby_dealers', 'comprehensive', or 'price_negotiation'",
            "prompt": "What type of dealer search? (contact_info, nearby_dealers, comprehensive, or price_negotiation)",
            "required": False,
        },
        {
            "name": "brands_interested",
            "type": "str",
            "description": "Specific car brands interested in (comma-separated, or 'any' for all brands)",
            "prompt": "Which car brands are you interested in? (e.g., 'Toyota, Honda' or 'any')",
            "required": False,
        },
        {
            "name": "max_distance",
            "type": "str",
            "description": "Maximum distance in miles to search (default: 50)",
            "prompt": "What's the maximum distance you're willing to travel (in miles)?",
            "required": False,
        },
        {
            "name": "services_needed",
            "type": "str",
            "description": "Required services (Sales, Service, Parts, Financing, Body_Shop)",
            "prompt": "What services do you need? (Sales, Service, Parts, Financing, Body_Shop)",
            "required": False,
        },
        {
            "name": "car_model",
            "type": "str",
            "description": "Car model for price negotiation analysis (required when search_type is 'price_negotiation')",
            "prompt": "Which car model are you negotiating for?",
            "required": False,
        },
        {
            "name": "asking_price",
            "type": "str",
            "description": "Asking price from dealer for negotiation analysis (USD)",
            "prompt": "What is the dealer's asking price?",
            "required": False,
        },
        {
            "name": "vehicle_condition",
            "type": "str",
            "description": "Vehicle condition for negotiation analysis (new, used, certified pre-owned)",
            "prompt": "What is the vehicle condition?",
            "required": False,
        }
    ],
    [
        {
            "name": "dealer_results",
            "type": "str",
            "description": "Comprehensive dealer information including contact details, locations, distances, services, and ratings in JSON format",
        }
    ],
)
def dealer_contact(location: str, search_type: str = "comprehensive", brands_interested: str = "any", max_distance: str = "50", services_needed: str = "Sales", car_model: str = "", asking_price: str = "", vehicle_condition: str = "new", **kwargs) -> str:
    """Comprehensive dealer finder with contact info, location data, and price negotiation analysis"""
    
    # Initialize variables for exception handler
    brands_filter = None
    
    try:
        logger.info(f"Searching for dealers near {location}, type: {search_type}, brands: {brands_interested}, max distance: {max_distance}")
        
        # Handle price negotiation search type
        if search_type == "price_negotiation":
            if not car_model or not asking_price:
                return json.dumps({
                    "status": "parameter_error",
                    "message": "Price negotiation analysis requires car_model and asking_price parameters",
                    "required_parameters": {
                        "car_model": "Specific car model to analyze for negotiation",
                        "asking_price": "Dealer's asking price in USD"
                    }
                }, indent=2, ensure_ascii=False)
            
            return _handle_price_negotiation(location, car_model, asking_price, vehicle_condition, brands_interested, max_distance)
        
        # Parse parameters for other search types
        max_dist = float(max_distance) if max_distance is not None else 50.0
        
        # Get user coordinates
        user_coords = _get_coordinates_for_location(location)
        if not user_coords:
            return json.dumps({
                "status": "location_error",
                "message": f"Could not find coordinates for location: {location}",
                "suggestions": [
                    "Try major cities like 'Los Angeles', 'New York', 'Chicago'",
                    "Use state names like 'California', 'Texas', 'Florida'",
                    "Use full city names like 'San Francisco' instead of 'SF'"
                ],
                "fallback_guidance": {
                    "search_suggestions": [
                        f"Search Google Maps for '{brands_interested + ' ' if brands_interested != 'any' else ''}car dealers near {location}'",
                        f"Visit manufacturer websites to find local dealers",
                        "Try major city names or state names for broader search"
                    ],
                    "general_tips": [
                        "Call dealers before visiting to confirm hours and availability",
                        "Ask about current promotions and incentives",
                        "Bring driver's license and proof of insurance for test drives"
                    ]
                }
            }, indent=2, ensure_ascii=False)
        
        user_lat, user_lon = user_coords
        
        # Parse brands filter
        if brands_interested.lower() == "any":
            brands_filter = None
        else:
            brands_filter = [brand.strip().title() for brand in brands_interested.split(",")]
        
        # Parse services filter
        required_services = [service.strip() for service in services_needed.split(",")]
        
        # Query dealers from database
        conn = _get_database_connection()
        cursor = conn.cursor()
        
        # Build dynamic query based on filters
        query = """
        SELECT 
            d.id, d.name, d.brand_affiliations, d.address_street, d.city, d.state, d.zip_code,
            d.phone, d.website, d.email, d.business_hours, d.services_offered,
            d.customer_rating, d.google_reviews_count, d.better_business_bureau_rating,
            d.volume_dealer, d.certified_pre_owned, d.loaner_vehicles, d.shuttle_service,
            d.saturday_service, d.languages_spoken, d.finance_partners, d.special_programs,
            d.inventory_size, d.years_in_business, d.latitude, d.longitude,
            COUNT(i.id) as available_inventory
        FROM dealers d
        LEFT JOIN inventory i ON d.id = i.dealer_id AND i.status = 'Available'
        WHERE 1=1
        """
        
        params = []
        
        # Add brand filter if specified
        if brands_filter:
            brand_conditions = []
            for brand in brands_filter:
                brand_conditions.append("d.brand_affiliations LIKE ?")
                params.append(f'%"{brand}"%')
            query += f" AND ({' OR '.join(brand_conditions)})"
        
        # Add services filter
        if required_services and required_services != ['']:
            for service in required_services:
                if service.strip():
                    query += " AND d.services_offered LIKE ?"
                    params.append(f"%{service.strip()}%")
        
        query += " GROUP BY d.id ORDER BY d.customer_rating DESC"
        
        cursor.execute(query, params)
        dealers = cursor.fetchall()
        
        if not dealers:
            conn.close()
            return json.dumps({
                "status": "no_dealers",
                "message": f"No {' '.join(brands_filter) + ' ' if brands_filter else ''}dealers found in our database",
                "location_searched": location,
                "suggestions": [
                    "Try searching without brand filter",
                    "Expand to nearby cities or states", 
                    "Contact manufacturer customer service for dealer locations"
                ],
                "fallback_options": [
                    f"Search Google for '{' '.join(brands_filter) + ' ' if brands_filter else ''}dealers near {location}'",
                    "Visit manufacturer websites for dealer locators",
                    "Contact manufacturer customer service"
                ]
            }, indent=2, ensure_ascii=False)
        
        # Calculate distances and filter by max distance
        dealer_results = []
        
        for dealer in dealers:
            dealer_lat, dealer_lon = dealer[24], dealer[25]  # latitude, longitude
            
            if dealer_lat and dealer_lon:
                distance = _calculate_distance(user_lat, user_lon, dealer_lat, dealer_lon)
                
                # Filter by max distance
                if distance <= max_dist:
                    dealer_info = _format_dealer_information(dealer, distance, search_type)
                    dealer_results.append(dealer_info)
            else:
                # Include dealers without coordinates but mark distance as unavailable
                dealer_info = _format_dealer_information(dealer, None, search_type)
                dealer_results.append(dealer_info)
        
        # Sort by distance first, then by rating
        dealer_results.sort(key=lambda x: (
            x.get("location_details", {}).get("distance_miles", 999),
            -(x.get("reputation", {}).get("customer_rating", 0) or 0)
        ))
        
        # Limit results to top 10
        dealer_results = dealer_results[:10]
        
        conn.close()
        
        # Generate response based on search type
        if search_type == "contact_info":
            result = _generate_contact_focused_response(dealer_results, location, brands_filter, services_needed)
        elif search_type == "nearby_dealers":
            result = _generate_location_focused_response(dealer_results, location, brands_filter, max_dist, user_lat, user_lon)
        else:  # comprehensive
            result = _generate_comprehensive_response(dealer_results, location, brands_filter, max_dist, services_needed, user_lat, user_lon)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in dealer search: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to search dealers: {str(e)}",
            "fallback_options": [
                f"Try Google search for '{' '.join(brands_filter) + ' ' if brands_filter else ''}dealers near {location}'",
                "Visit manufacturer websites for dealer locators",
                "Contact manufacturer customer service"
            ],
            "suggestions": [
                "Try a different location format",
                "Ensure the car database is properly initialized",
                "Contact support if the issue persists"
            ]
        }, indent=2, ensure_ascii=False)

def _format_dealer_information(dealer_data: tuple, distance: Optional[float], search_type: str) -> Dict[str, Any]:
    """Format dealer information based on search type"""
    
    # Parse JSON fields safely
    try:
        brand_affiliations = json.loads(dealer_data[2]) if dealer_data[2] else []
    except:
        brand_affiliations = [dealer_data[2]] if dealer_data[2] else []
    
    try:
        business_hours = json.loads(dealer_data[10]) if dealer_data[10] else {}
    except:
        business_hours = {"Note": dealer_data[10]} if dealer_data[10] else {}
    
    try:
        finance_partners = json.loads(dealer_data[21]) if dealer_data[21] else []
    except:
        finance_partners = [dealer_data[21]] if dealer_data[21] else []
    
    try:
        special_programs = json.loads(dealer_data[22]) if dealer_data[22] else []
    except:
        special_programs = [dealer_data[22]] if dealer_data[22] else []
    
    # Base information all search types need
    base_info = {
        "dealer_id": dealer_data[0],
        "name": dealer_data[1],
        "brands_sold": brand_affiliations,
        "contact_information": {
            "phone": dealer_data[7],
            "website": dealer_data[8],
            "email": dealer_data[9],
            "address": {
                "street": dealer_data[3],
                "city": dealer_data[4],
                "state": dealer_data[5],
                "zip_code": dealer_data[6],
                "full_address": f"{dealer_data[3]}, {dealer_data[4]}, {dealer_data[5]} {dealer_data[6]}"
            }
        }
    }
    
    # Add detailed information based on search type
    if search_type == "contact_info":
        base_info.update({
            "business_information": {
                "hours": business_hours,
                "services_offered": dealer_data[11].split(",") if dealer_data[11] else [],
                "languages_spoken": dealer_data[20].split(",") if dealer_data[20] else ["English"],
                "years_in_business": dealer_data[24],
                "established_year": 2024 - dealer_data[24] if dealer_data[24] else "Unknown"
            },
            "contact_recommendations": _generate_contact_recommendations(dealer_data)
        })
    
    elif search_type in ["nearby_dealers", "comprehensive"]:
        base_info.update({
            "business_information": {
                "hours": business_hours,
                "services_offered": dealer_data[11].split(",") if dealer_data[11] else [],
                "languages_spoken": dealer_data[20].split(",") if dealer_data[20] else ["English"],
                "years_in_business": dealer_data[24],
                "established_year": 2024 - dealer_data[24] if dealer_data[24] else "Unknown"
            },
            "reputation": {
                "customer_rating": dealer_data[12],
                "google_reviews_count": dealer_data[13],
                "better_business_bureau_rating": dealer_data[14],
                "volume_dealer": bool(dealer_data[15])
            },
            "services_and_amenities": {
                "certified_pre_owned": bool(dealer_data[16]),
                "loaner_vehicles": bool(dealer_data[17]),
                "shuttle_service": bool(dealer_data[18]),
                "saturday_service": bool(dealer_data[19]),
                "financing_partners": finance_partners,
                "special_programs": special_programs
            },
            "inventory_information": {
                "total_inventory_size": dealer_data[23],
                "currently_available": dealer_data[26]
            },
            "location_details": {
                "distance_miles": distance if distance is not None else "Distance unavailable",
                "coordinates": {
                    "latitude": dealer_data[24],
                    "longitude": dealer_data[25]
                }
            }
        })
    
    if search_type == "comprehensive":
        base_info["contact_recommendations"] = _generate_contact_recommendations(dealer_data)
    
    return base_info

def _generate_contact_focused_response(dealers: List[Dict], location: str, brands_filter: Optional[List[str]], services_needed: str) -> Dict[str, Any]:
    """Generate response focused on contact information"""
    return {
        "status": "success",
        "search_type": "contact_info",
        "search_summary": {
            "location_searched": location,
            "brand_filter": brands_filter if brands_filter else "All brands",
            "services_filter": services_needed,
            "total_dealers_found": len(dealers),
            "search_timestamp": datetime.now().isoformat()
        },
        "dealer_contacts": dealers,
        "contact_best_practices": {
            "before_calling": [
                "Check their business hours and preferred contact methods",
                "Prepare your questions about specific vehicles or services",
                "Have your driver's license and insurance ready for test drives",
                "Know your budget range and financing preferences"
            ],
            "when_calling": [
                "Ask to speak with a sales consultant or service advisor",
                "Mention any specific vehicles you're interested in",
                "Ask about current promotions and manufacturer incentives",
                "Inquire about appointment availability"
            ],
            "what_to_ask": [
                "Current inventory and vehicle availability",
                "Pricing and any current rebates or incentives",
                "Financing options and interest rates",
                "Trade-in evaluation process",
                "Service department hours and scheduling",
                "Warranty and maintenance programs"
            ]
        },
        "next_steps": [
            "Call ahead to confirm inventory and schedule appointments",
            "Visit dealer websites to browse current inventory",
            "Check online reviews and ratings",
            "Compare financing offers from multiple dealers",
            "Prepare all necessary documents for purchase"
        ]
    }

def _generate_location_focused_response(dealers: List[Dict], location: str, brands_filter: Optional[List[str]], max_distance: float, user_lat: float, user_lon: float) -> Dict[str, Any]:
    """Generate response focused on location and proximity"""
    return {
        "status": "success",
        "search_type": "nearby_dealers",
        "search_criteria": {
            "location": location,
            "coordinates": {"latitude": user_lat, "longitude": user_lon},
            "brands_filter": brands_filter if brands_filter else "All brands",
            "max_distance_miles": max_distance
        },
        "results_summary": {
            "total_dealers_found": len(dealers),
            "search_radius_miles": max_distance,
            "closest_dealer_distance": dealers[0]["location_details"]["distance_miles"] if dealers and dealers[0]["location_details"]["distance_miles"] != "Distance unavailable" else None
        },
        "nearby_dealers": dealers,
        "location_recommendations": _generate_dealer_recommendations(dealers),
        "next_steps": [
            "Call ahead to confirm inventory availability",
            "Schedule an appointment for best service",
            "Ask about current promotions and incentives",
            "Verify financing options if needed",
            "Bring driver's license and proof of insurance for test drives"
        ]
    }

def _generate_comprehensive_response(dealers: List[Dict], location: str, brands_filter: Optional[List[str]], max_distance: float, services_needed: str, user_lat: float, user_lon: float) -> Dict[str, Any]:
    """Generate comprehensive response with all information"""
    market_summary = _generate_dealer_summary(dealers, location, ' '.join(brands_filter) if brands_filter else "")
    
    return {
        "status": "success",
        "search_type": "comprehensive",
        "search_summary": {
            "location_searched": location,
            "coordinates": {"latitude": user_lat, "longitude": user_lon},
            "brand_filter": brands_filter if brands_filter else "All brands",
            "services_filter": services_needed,
            "max_distance_miles": max_distance,
            "total_dealers_found": len(dealers),
            "search_timestamp": datetime.now().isoformat()
        },
        "results_summary": {
            "total_dealers_found": len(dealers),
            "search_radius_miles": max_distance,
            "closest_dealer_distance": dealers[0]["location_details"]["distance_miles"] if dealers and dealers[0]["location_details"]["distance_miles"] != "Distance unavailable" else None
        },
        "dealers": dealers,
        "market_analysis": market_summary,
        "recommendations": _generate_dealer_recommendations(dealers),
        "contact_best_practices": {
            "before_calling": [
                "Check their business hours and preferred contact methods",
                "Prepare your questions about specific vehicles or services",
                "Have your driver's license and insurance ready for test drives",
                "Know your budget range and financing preferences"
            ],
            "when_calling": [
                "Ask to speak with a sales consultant or service advisor",
                "Mention any specific vehicles you're interested in",
                "Ask about current promotions and manufacturer incentives",
                "Inquire about appointment availability"
            ],
            "what_to_ask": [
                "Current inventory and vehicle availability",
                "Pricing and any current rebates or incentives",
                "Financing options and interest rates",
                "Trade-in evaluation process",
                "Service department hours and scheduling",
                "Warranty and maintenance programs"
            ]
        },
        "next_steps": [
            "Call ahead to confirm inventory and schedule appointments",
            "Visit dealer websites to browse current inventory",
            "Check online reviews and ratings",
            "Compare financing offers from multiple dealers",
            "Prepare all necessary documents for purchase",
            "Consider visiting multiple dealers for price comparison"
        ]
    }

def _generate_contact_recommendations(dealer_data: tuple) -> Dict[str, str]:
    """Generate personalized contact recommendations for each dealer"""
    name = dealer_data[1]
    rating = dealer_data[12]
    services = dealer_data[11]
    saturday_service = dealer_data[19]
    shuttle_service = dealer_data[18]
    
    recommendations = {
        "best_contact_method": "Phone call for immediate response, email for detailed inquiries",
        "optimal_timing": "Weekday mornings (9-11 AM) or after lunch (2-4 PM)",
        "special_notes": []
    }
    
    if rating and rating >= 4.5:
        recommendations["special_notes"].append("Highly rated dealer - expect excellent customer service")
    
    if saturday_service:
        recommendations["optimal_timing"] = "Weekdays or Saturdays - they offer weekend service"
    
    if shuttle_service:
        recommendations["special_notes"].append("Offers shuttle service - convenient for service appointments")
    
    if "Financing" in (services or ""):
        recommendations["special_notes"].append("Full financing services available - ask about rates and terms")
    
    return recommendations

def _generate_dealer_recommendations(dealers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate intelligent recommendations based on dealer data"""
    if not dealers:
        return {"message": "No dealers found in the specified area"}
    
    recommendations = {}
    
    # Best rated dealer
    dealers_with_rating = [d for d in dealers if d.get("reputation", {}).get("customer_rating")]
    if dealers_with_rating:
        best_rated = max(dealers_with_rating, key=lambda x: x["reputation"]["customer_rating"] or 0)
        recommendations["highest_rated"] = {
            "name": best_rated["name"],
            "rating": best_rated["reputation"]["customer_rating"],
            "reason": f"Highest customer rating with {best_rated['reputation']['google_reviews_count']} Google reviews"
        }
    
    # Closest dealer
    dealers_with_distance = [d for d in dealers if d.get("location_details", {}).get("distance_miles") != "Distance unavailable"]
    if dealers_with_distance:
        closest = min(dealers_with_distance, key=lambda x: x["location_details"]["distance_miles"])
        recommendations["closest"] = {
            "name": closest["name"],
            "distance": f"{closest['location_details']['distance_miles']} miles",
            "reason": "Shortest travel distance"
        }
    
    # Best inventory
    dealers_with_inventory = [d for d in dealers if d.get("inventory_information", {}).get("currently_available", 0) > 0]
    if dealers_with_inventory:
        best_inventory = max(dealers_with_inventory, key=lambda x: x["inventory_information"]["currently_available"] or 0)
        recommendations["best_inventory"] = {
            "name": best_inventory["name"],
            "available_units": best_inventory["inventory_information"]["currently_available"],
            "reason": "Largest selection of available vehicles"
        }
    
    # Most services
    most_services = max(dealers, key=lambda x: len(x.get("business_information", {}).get("services_offered", [])))
    recommendations["full_service"] = {
        "name": most_services["name"],
        "services": most_services.get("business_information", {}).get("services_offered", []),
        "reason": "Offers the most comprehensive services"
    }
    
    # Best for financing
    financing_dealers = [d for d in dealers if d.get("services_and_amenities", {}).get("financing_partners")]
    if financing_dealers:
        best_financing = max(financing_dealers, key=lambda x: len(x.get("services_and_amenities", {}).get("financing_partners", [])))
        recommendations["best_financing"] = {
            "name": best_financing["name"],
            "partners": len(best_financing["services_and_amenities"]["financing_partners"]),
            "programs": best_financing["services_and_amenities"]["special_programs"],
            "reason": "Most financing options and special programs"
        }
    
    return recommendations

def _generate_dealer_summary(dealers: List[Dict], location: str, brand: str) -> Dict[str, Any]:
    """Generate market summary and insights"""
    if not dealers:
        return {}
    
    # Calculate averages and insights
    dealers_with_rating = [d for d in dealers if d.get("reputation", {}).get("customer_rating")]
    if dealers_with_rating:
        avg_rating = sum(d["reputation"]["customer_rating"] for d in dealers_with_rating) / len(dealers_with_rating)
    else:
        avg_rating = 0
    
    total_inventory = sum(d.get("inventory_information", {}).get("currently_available", 0) for d in dealers)
    
    service_counts = {}
    for dealer in dealers:
        for service in dealer.get("business_information", {}).get("services_offered", []):
            service_counts[service] = service_counts.get(service, 0) + 1
    
    top_services = sorted(service_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    weekend_service_count = len([d for d in dealers if d.get("services_and_amenities", {}).get("saturday_service")])
    shuttle_service_count = len([d for d in dealers if d.get("services_and_amenities", {}).get("shuttle_service")])
    
    return {
        "market_overview": {
            "average_dealer_rating": round(avg_rating, 1) if avg_rating > 0 else "No ratings available",
            "total_available_inventory": total_inventory,
            "dealers_with_weekend_service": weekend_service_count,
            "dealers_with_shuttle_service": shuttle_service_count
        },
        "most_common_services": [{"service": service, "dealer_count": count} for service, count in top_services],
        "market_insights": [
            f"Found {len(dealers)} {brand + ' ' if brand else ''}dealers in the {location} area",
            f"Average customer rating is {avg_rating:.1f}/5.0" if avg_rating > 0 else "Rating information varies by dealer",
            f"Total of {total_inventory} vehicles currently available across all dealers",
            f"{weekend_service_count} dealers offer Saturday service",
            f"{shuttle_service_count} dealers provide shuttle service"
        ]
    }
# Price Negotiation Functions (integrated from price_negotiate.py)

def _handle_price_negotiation(location: str, car_model: str, asking_price: str, vehicle_condition: str, brands_interested: str, max_distance: str) -> str:
    """Handle price negotiation analysis with dealer recommendations"""
    try:
        logger.info(f"Analyzing negotiation for: {car_model}, asking price: {asking_price}")
        
        # Parse asking price
        try:
            price = float(asking_price.replace("$", "").replace(",", ""))
        except ValueError:
            return json.dumps({
                "status": "error",
                "message": "Invalid price format. Please provide a numeric value."
            }, indent=2, ensure_ascii=False)
        
        # Generate negotiation analysis
        analysis = _generate_negotiation_analysis(car_model, price, vehicle_condition)
        
        # Get dealer recommendations for this area and brand
        user_coords = _get_coordinates_for_location(location)
        dealer_recommendations = []
        
        if user_coords:
            # Extract brand from car model for dealer search
            model_words = car_model.lower().split()
            potential_brands = ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes", "Audi", "Lexus", "Acura", "Infiniti", "Cadillac", "Lincoln", "Buick", "GMC", "Dodge", "Chrysler", "Jeep", "Ram", "Nissan", "Hyundai", "Kia", "Subaru", "Mazda", "Mitsubishi", "Volkswagen", "Volvo", "Jaguar", "Land Rover", "Porsche", "Ferrari", "Lamborghini", "Maserati", "Bentley", "Rolls-Royce", "Aston Martin", "McLaren", "Tesla", "Rivian", "Lucid"]
            
            detected_brand = None
            for brand in potential_brands:
                if brand.lower() in model_words or any(brand.lower() in word for word in model_words):
                    detected_brand = brand
                    break
            
            if detected_brand:
                try:
                    # Get dealers for this brand
                    conn = _get_database_connection()
                    cursor = conn.cursor()
                    
                    query = """
                    SELECT name, phone, customer_rating, address_street, city, state, zip_code, latitude, longitude
                    FROM dealers 
                    WHERE brand_affiliations LIKE ? AND services_offered LIKE "%Sales%"
                    ORDER BY customer_rating DESC LIMIT 5
                    """
                    
                    cursor.execute(query, [f"%\"{detected_brand}\"%"])
                    dealers = cursor.fetchall()
                    
                    user_lat, user_lon = user_coords
                    for dealer in dealers:
                        if dealer[7] and dealer[8]:  # has coordinates
                            distance = _calculate_distance(user_lat, user_lon, dealer[7], dealer[8])
                            dealer_recommendations.append({
                                "name": dealer[0],
                                "phone": dealer[1],
                                "rating": dealer[2],
                                "address": f"{dealer[3]}, {dealer[4]}, {dealer[5]} {dealer[6]}",
                                "distance_miles": distance
                            })
                    
                    conn.close()
                    
                    # Sort by distance
                    dealer_recommendations.sort(key=lambda x: x["distance_miles"])
                    dealer_recommendations = dealer_recommendations[:3]  # Top 3 closest
                    
                except Exception as e:
                    logger.warning(f"Could not get dealer recommendations: {e}")
        
        result = {
            "status": "success",
            "search_type": "price_negotiation",
            "car_model": car_model,
            "vehicle_condition": vehicle_condition,
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "negotiation_analysis": analysis,
            "recommended_dealers": dealer_recommendations,
            "next_steps": [
                "Use the negotiation strategies with local dealers",
                "Get quotes from multiple dealers for comparison",
                "Research competing offers before negotiations",
                "Time your visit for maximum leverage (end of month/quarter)",
                "Prepare to walk away if terms are not acceptable"
            ]
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error analyzing negotiation: {str(e)}")
        error_result = {
            "status": "error",
            "message": f"Analysis failed: {str(e)}"
        }
        return json.dumps(error_result, indent=2, ensure_ascii=False)


def _generate_negotiation_analysis(car_model: str, price: float, condition: str) -> Dict[str, Any]:
    """Generate comprehensive negotiation analysis"""
    
    # Determine negotiation potential based on price range and condition
    negotiation_potential = _assess_negotiation_potential(price, condition)
    
    # Generate price analysis
    price_analysis = _analyze_pricing_strategy(price, condition)
    
    # Get negotiation strategies
    strategies = _get_negotiation_strategies(condition, price)
    
    # Get timing advice
    timing_advice = _get_timing_advice()
    
    # Get preparation checklist
    preparation = _get_preparation_checklist()
    
    return {
        "price_analysis": price_analysis,
        "negotiation_potential": negotiation_potential,
        "strategies": strategies,
        "timing_advice": timing_advice,
        "preparation_checklist": preparation,
        "red_flags": _get_negotiation_red_flags(),
        "closing_tips": _get_closing_tips()
    }

def _assess_negotiation_potential(price: float, condition: str) -> Dict[str, Any]:
    """Assess negotiation potential based on price and condition"""
    
    potential_savings = 0
    flexibility = "Medium"
    
    if condition.lower() == "new":
        if price > 60000:  # Luxury vehicles
            potential_savings = price * 0.08  # 8% potential savings
            flexibility = "High"
        elif price > 35000:  # Mid-range vehicles
            potential_savings = price * 0.05  # 5% potential savings
            flexibility = "Medium"
        else:  # Economy vehicles
            potential_savings = price * 0.03  # 3% potential savings
            flexibility = "Low"
    elif condition.lower() in ["used", "certified pre-owned"]:
        potential_savings = price * 0.12  # Higher negotiation room for used cars
        flexibility = "High"
    
    return {
        "estimated_savings_range": f"${potential_savings * 0.5:,.0f} - ${potential_savings:,.0f}",
        "negotiation_flexibility": flexibility,
        "success_probability": "70-85%" if flexibility == "High" else "50-70%" if flexibility == "Medium" else "30-50%"
    }

def _analyze_pricing_strategy(price: float, condition: str) -> Dict[str, Any]:
    """Analyze pricing strategy recommendations"""
    
    # Calculate target prices
    if condition.lower() == "new":
        target_discount = 0.05 if price > 50000 else 0.03
    else:
        target_discount = 0.10
    
    target_price = price * (1 - target_discount)
    walk_away_price = price * (1 - target_discount/2)
    
    return {
        "asking_price": f"${price:,.0f}",
        "target_price": f"${target_price:,.0f}",
        "walk_away_price": f"${walk_away_price:,.0f}",
        "opening_offer": f"${target_price * 0.95:,.0f}",
        "negotiation_range": f"${target_price:,.0f} - ${walk_away_price:,.0f}"
    }

def _get_negotiation_strategies(condition: str, price: float) -> List[Dict[str, str]]:
    """Get condition-specific negotiation strategies"""
    
    base_strategies = [
        {
            "strategy": "Research Market Prices",
            "description": "Compare prices from multiple dealers and online sources",
            "effectiveness": "High",
            "timing": "Before visiting dealer"
        },
        {
            "strategy": "Get Multiple Quotes",
            "description": "Obtain written quotes from 3-4 dealers for comparison",
            "effectiveness": "High",
            "timing": "Before negotiating"
        },
        {
            "strategy": "Negotiate Out-the-Door Price",
            "description": "Focus on total price including all fees, not monthly payments",
            "effectiveness": "High",
            "timing": "During negotiation"
        },
        {
            "strategy": "Separate Trade-in Discussion",
            "description": "Negotiate car price first, then discuss trade-in separately",
            "effectiveness": "Medium",
            "timing": "During negotiation"
        }
    ]
    
    if condition.lower() == "new":
        base_strategies.extend([
            {
                "strategy": "End-of-Month/Quarter Timing",
                "description": "Shop at month/quarter end when dealers need to meet quotas",
                "effectiveness": "Medium",
                "timing": "Strategic timing"
            },
            {
                "strategy": "Factory Incentives",
                "description": "Research manufacturer rebates and incentives",
                "effectiveness": "High",
                "timing": "Before negotiating"
            }
        ])
    else:
        base_strategies.extend([
            {
                "strategy": "Vehicle History Report",
                "description": "Use any issues found as negotiation leverage",
                "effectiveness": "Medium",
                "timing": "After inspection"
            },
            {
                "strategy": "Inspection Leverage",
                "description": "Point out needed repairs or maintenance",
                "effectiveness": "High",
                "timing": "After inspection"
            }
        ])
    
    return base_strategies

def _get_timing_advice() -> Dict[str, Any]:
    """Get timing advice for negotiations"""
    
    return {
        "best_times": [
            "End of month when dealers need to meet quotas",
            "End of model year for outgoing models",
            "Weekdays when salespeople have more time",
            "Late in the day when dealers want to close deals"
        ],
        "avoid_times": [
            "First day of the month when inventory is fresh",
            "Weekends when dealers are busy",
            "Beginning of model year for new releases"
        ],
        "seasonal_factors": {
            "winter": "Good time for convertibles and sports cars",
            "spring": "SUV and truck demand increases",
            "summer": "High demand season, less negotiation room",
            "fall": "Good time as new models arrive"
        }
    }

def _get_preparation_checklist() -> List[str]:
    """Get preparation checklist for negotiations"""
    
    return [
        "Research market value and competitor prices",
        "Get pre-approved for financing from bank/credit union",
        "Determine your maximum budget and stick to it",
        "Gather trade-in documentation and get independent appraisal",
        "Review your credit score and report",
        "Prepare list of must-have vs nice-to-have features",
        "Plan your negotiation strategy and talking points",
        "Bring a friend for emotional support and second opinion",
        "Set aside full day for the process",
        "Prepare to walk away if terms are not acceptable"
    ]

def _get_negotiation_red_flags() -> List[str]:
    """Get list of negotiation red flags to watch for"""
    
    return [
        "Pressure to 'buy today' or limited-time offers",
        "Refusal to provide written quotes or estimates",
        "Adding unexpected fees or charges",
        "Focusing only on monthly payments, not total price",
        "Unwillingness to let you inspect vehicle thoroughly",
        "Aggressive or disrespectful behavior",
        "Bait-and-switch tactics on advertised vehicles",
        "Requiring immediate decision without time to think",
        "Hiding or rushing through paperwork",
        "Unusual financing terms or high interest rates"
    ]

def _get_closing_tips() -> List[str]:
    """Get tips for closing the deal"""
    
    return [
        "Read all paperwork carefully before signing",
        "Verify all agreed-upon terms are in writing",
        "Double-check all fees and charges",
        "Confirm delivery date and vehicle preparation",
        "Understand warranty terms and coverage",
        "Get copies of all signed documents",
        "Verify financing terms match what was discussed",
        "Confirm trade-in value and payoff process",
        "Schedule delivery appointment if needed",
        "Don't sign anything you don't understand"
    ]
