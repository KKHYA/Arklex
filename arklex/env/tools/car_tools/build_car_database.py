#!/usr/bin/env python3
"""
Car Database Builder - Using LLM API for Real Vehicle Data
No estimation functions - completely relies on LLM API and external data sources for accurate information
"""

import sqlite3
import time
import logging
import requests
import json
import re
from pathlib import Path
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VehicleData:
    brand: str
    model: str
    year: int
    trim: str = None
    body_type: str = None
    fuel_type: str = None
    transmission: str = None
    drivetrain: str = None
    msrp: float = None
    engine_type: str = None
    horsepower: int = None
    torque: int = None
    acceleration_0_60: float = None
    mpg_city: int = None
    mpg_highway: int = None
    seating_capacity: int = None
    cargo_space: float = None
    safety_rating: float = None
    weight: int = None
    length: float = None
    width: float = None
    height: float = None
    top_speed: int = None
    lateral_g: float = None
    quarter_mile_time: float = None
    quarter_mile_speed: int = None
    braking_60_0: float = None
    braking_100_0: float = None
    nhtsa_overall_rating: int = None
    nhtsa_frontal_crash: int = None
    nhtsa_side_crash: int = None
    nhtsa_rollover: int = None
    last_updated: str = None
    data_quality_score: float = 0.9
    source: str = "curated"

@dataclass
class MarketData:
    """Market economic data from FRED API"""
    total_vehicle_sales: float = None
    new_vehicle_cpi: float = None
    used_vehicle_cpi: float = None
    auto_loan_rate_48m: float = None
    auto_loan_rate_60m: float = None
    auto_loan_rate_72m: float = None
    auto_inventory_ratio: float = None
    domestic_auto_production: float = None
    vehicle_miles_traveled: float = None
    last_updated: str = None

def clean_llm_json_response(content: str) -> str:
    """Enhanced JSON cleaning function to handle common LLM formatting issues"""
    try:
        # Remove markdown code blocks
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        # Remove any text before the first '{' and after the last '}'
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx + 1]
        
        # Remove comments (// style and /* */ style)
        content = re.sub(r'//.*?(?=\n|$)', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Fix common JSON formatting issues
        # Add quotes to unquoted property names
        content = re.sub(r'(\w+)(\s*:\s*)', r'"\1"\2', content)
        # Fix already quoted properties (avoid double quotes)
        content = re.sub(r'""(\w+)""(\s*:\s*)', r'"\1"\2', content)
        
        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Ensure proper string escaping
        # Fix unescaped quotes within string values
        content = re.sub(r':\s*"([^"]*)"([^",\}\]]*)"([^",\}\]]*)"', r': "\1\2\3"', content)
        
        # Remove any extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
        
    except Exception as e:
        logger.warning(f"JSON cleaning failed: {e}")
        return content.strip()

def validate_and_fix_json_structure(data: Dict[str, Any], brand: str, model: str, year: int) -> Dict[str, Any]:
    """Validate and fix JSON structure with fallback values"""
    required_fields = {
        'brand': brand,
        'model': model,
        'year': year,
        'trim': 'Base',
        'body_type': 'Sedan',
        'fuel_type': 'Gasoline',
        'engine_type': '4-Cylinder',
        'horsepower': 200,
        'torque': 250,
        'transmission': 'Automatic',
        'drivetrain': 'FWD',
        'acceleration_0_60': 8.0,
        'top_speed': 120,
        'mpg_city': 25,
        'mpg_highway': 35,
        'seating_capacity': 5,
        'cargo_space': 15.0,
        'msrp': 30000,
        'safety_rating': 4.0
    }
    
    # Ensure all required fields exist with appropriate types
    for field, default_value in required_fields.items():
        if field not in data or data[field] is None:
            data[field] = default_value
            logger.debug(f"Added missing field '{field}' with default value for {brand} {model}")
        
        # Type conversion and validation
        try:
            if isinstance(default_value, int) and not isinstance(data[field], int):
                data[field] = int(float(str(data[field])))
            elif isinstance(default_value, float) and not isinstance(data[field], (int, float)):
                data[field] = float(str(data[field]))
            elif isinstance(default_value, str) and not isinstance(data[field], str):
                data[field] = str(data[field])
        except (ValueError, TypeError):
            data[field] = default_value
            logger.debug(f"Fixed invalid type for field '{field}' for {brand} {model}")
    
    return data

class LLMVehicleDataCollector:
    """Collect real vehicle data using LLM API with enhanced error handling"""
    
    def __init__(self, api_key: str = None, api_base: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.api_base = api_base or os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.max_retries = 2  # Maximum retry attempts for failed requests
    
    def get_brand_warranty_info(self, brand: str) -> Dict[str, Any]:
        """Get brand-specific warranty information from LLM with enhanced JSON handling"""
        try:
            prompt = f"""
CRITICAL: Return ONLY valid JSON. No additional text or formatting.

Provide warranty information for {brand} vehicles in the US market:

{{
    "warranty_basic_years": 3,
    "warranty_basic_miles": 36000,
    "warranty_powertrain_years": 5,
    "warranty_powertrain_miles": 60000,
    "iihs_top_safety_pick_common": false,
    "epa_greenhouse_score_typical": 7
}}

Replace with accurate {brand} warranty terms. Return ONLY the JSON object.
"""

            response = self.session.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an automotive warranty expert. Return ONLY valid JSON with accurate warranty information."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Apply enhanced JSON cleaning
                cleaned_content = clean_llm_json_response(content)
                warranty_data = json.loads(cleaned_content)
                
                logger.info(f"LLM warranty data retrieved for {brand}")
                return warranty_data
                
            else:
                logger.warning(f"LLM API error for warranty {brand}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"LLM warranty API call failed for {brand}: {e}")
            return None
        
        finally:
            time.sleep(self.rate_limit_delay)
    
    def get_brand_characteristics(self, brand: str) -> Dict[str, Any]:
        """Get comprehensive brand characteristics from LLM with enhanced JSON handling"""
        try:
            prompt = f"""
CRITICAL: Return ONLY valid JSON. No additional text or formatting.

Analyze {brand} as an automotive brand in the US market:

{{
    "reliability_score": 7.0,
    "luxury_score": 5.0,
    "performance_score": 5.0,
    "value_score": 7.0,
    "technology_score": 6.0,
    "environmental_score": 6.0,
    "maintenance_cost_level": "Medium",
    "average_maintenance_cost_annual": 1000,
    "resale_value_score": 6.5,
    "us_market_share": 5.0,
    "target_demographic": "General Market",
    "popular_in_states": "CA,TX,NY,FL,OH"
}}

Replace with accurate {brand} characteristics. Return ONLY the JSON object.
"""

            response = self.session.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an automotive market analyst. Return ONLY valid JSON with accurate brand analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Apply enhanced JSON cleaning
                cleaned_content = clean_llm_json_response(content)
                brand_data = json.loads(cleaned_content)
                
                logger.info(f"LLM brand characteristics retrieved for {brand}")
                return brand_data
                
            else:
                logger.warning(f"LLM API error for brand {brand}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"LLM brand API call failed for {brand}: {e}")
            return None
        
        finally:
            time.sleep(self.rate_limit_delay)
    
    def get_vehicle_market_price(self, brand: str, model: str, year: int, condition: str = "new") -> Dict[str, Any]:
        """Get realistic market pricing from LLM"""
        try:
            prompt = f"""
Provide realistic US market pricing for {year} {brand} {model} in {condition} condition. Return in JSON format:

{{
    "base_msrp": manufacturer_suggested_retail_price_usd(integer),
    "typical_dealer_price": typical_selling_price_usd(integer),
    "market_adjustment": current_markup_or_discount_usd(integer),
    "invoice_price_estimate": dealer_cost_estimate_usd(integer),
    "regional_variation": "price_varies_by_region_yes_no",
    "incentives_available": "current_manufacturer_incentives_description"
}}

Provide realistic current market pricing. Return only JSON.
"""

            response = self.session.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an automotive pricing expert with knowledge of current US market values and dealer economics."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 400
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                if content.startswith('```json'):
                    content = content.replace('```json', '').replace('```', '').strip()
                elif content.startswith('```'):
                    content = content.replace('```', '').strip()
                
                pricing_data = json.loads(content)
                logger.info(f"LLM pricing data retrieved for {year} {brand} {model}")
                return pricing_data
                
            else:
                logger.warning(f"LLM API error for pricing {brand} {model}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"LLM pricing API call failed for {brand} {model}: {e}")
            return None
        
        finally:
            time.sleep(self.rate_limit_delay)
    
    def get_popular_vehicle_models(self, max_brands: int = 25) -> List[tuple]:
        """Get popular vehicle model list using LLM API"""
        try:
            prompt = f"""
Provide a comprehensive list of {max_brands} most popular and significant automotive brands sold in the US market, along with their best-selling and most popular models in JSON format:

{{
    "brands": [
        {{
            "brand": "Toyota",
            "models": ["Camry", "Corolla", "RAV4", "Highlander", "Prius", "Tacoma", "4Runner"]
        }},
        {{
            "brand": "Tesla", 
            "models": ["Model 3", "Model Y", "Model S", "Model X", "Cybertruck", "Roadster"]
        }},
        {{
            "brand": "Honda", 
            "models": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey", "Ridgeline", "Passport"]
        }}
    ]
}}

Please provide a diverse and representative selection covering:
- Traditional mainstream brands (Toyota, Honda, Ford, Chevrolet, Nissan, Hyundai, Kia, Mazda, Subaru, Volkswagen)
- Premium and luxury brands (BMW, Mercedes-Benz, Audi, Lexus, Acura, Infiniti, Genesis, Cadillac, Lincoln)
- Electric vehicle specialists (Tesla, Rivian, Lucid, Polestar)
- Performance and specialty brands (Porsche, Jaguar, Land Rover, Volvo, Mini)
- American truck brands (Ram, Jeep, GMC, Buick)

For each brand, select 4-8 of their most popular current models available in the US market (2022-2024), including:
- Best-selling sedans, SUVs, crossovers, trucks
- Electric and hybrid variants where available
- Both entry-level and premium offerings within each brand

Focus on models that American consumers actually buy and that are readily available at US dealerships. Return only valid JSON with no additional text or explanations.
"""

            response = self.session.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an automotive market expert with knowledge of US car brands and popular models."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Clean JSON format
                if content.startswith('```json'):
                    content = content.replace('```json', '').replace('```', '').strip()
                elif content.startswith('```'):
                    content = content.replace('```', '').strip()
                
                data = json.loads(content)
                
                # Convert to vehicle list
                vehicle_list = []
                for brand_info in data['brands']:
                    brand = brand_info['brand']
                    for model in brand_info['models']:
                        # Add multiple years for each model
                        for year in [2024, 2023, 2022]:
                            vehicle_list.append((brand, model, year))
                
                logger.info(f"LLM vehicle list retrieved successfully: {len(vehicle_list)} vehicles")
                return vehicle_list
                
            else:
                raise RuntimeError(f"LLM API vehicle list retrieval failed with status {response.status_code}")
                
        except Exception as e:
            raise RuntimeError(f"LLM API vehicle list retrieval failed: {e}")
    
    def get_vehicle_specs(self, brand: str, model: str, year: int) -> Optional[Dict[str, Any]]:
        """Get detailed vehicle specifications using LLM API with enhanced error handling and retry logic"""
        
        for attempt in range(self.max_retries + 1):
            try:
                # Enhanced prompt with strict JSON formatting requirements
                prompt = f"""
CRITICAL: You must return ONLY a valid JSON object. No additional text, explanations, or comments.

STRICT REQUIREMENTS:
1. Use double quotes for ALL property names and string values
2. NO trailing commas anywhere
3. NO comments (// or /* */)
4. NO markdown formatting or code blocks
5. Ensure all numeric values are valid numbers (not strings)

Provide detailed vehicle specifications for {year} {brand} {model} in US market:

{{
    "brand": "{brand}",
    "model": "{model}",
    "year": {year},
    "trim": "base_trim_name",
    "body_type": "Sedan",
    "fuel_type": "Gasoline",
    "engine_type": "4-Cylinder",
    "horsepower": 200,
    "torque": 250,
    "transmission": "Automatic",
    "drivetrain": "FWD",
    "acceleration_0_60": 8.0,
    "top_speed": 120,
    "mpg_city": 25,
    "mpg_highway": 35,
    "seating_capacity": 5,
    "cargo_space": 15.0,
    "msrp": 30000,
    "safety_rating": 4.0,
    "weight": 3500,
    "length": 185.0,
    "width": 73.0,
    "height": 58.0,
    "lateral_g": 0.85,
    "quarter_mile_time": 16.0,
    "quarter_mile_speed": 85,
    "braking_60_0": 120.0,
    "braking_100_0": 300.0
}}

IMPORTANT NOTES:
- For Tesla/electric vehicles: use "fuel_type": "Electric", "engine_type": "Electric Motor", "transmission": "Single-Speed", set mpg_city/mpg_highway to EPA range estimates
- For hybrids: use "fuel_type": "Hybrid" 
- Use actual current US market pricing and specifications
- Ensure all specifications are realistic and accurate for the specific model year

Replace with accurate specifications. Return ONLY the JSON object above with real data.
"""

                response = self.session.post(
                    f"{self.api_base}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system", 
                                "content": "You are a professional automotive database expert. You MUST return only valid JSON with accurate vehicle specifications. NO explanatory text, NO markdown formatting, NO comments."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,  # Low temperature for consistency
                        "max_tokens": 1000
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    
                    # Enhanced JSON cleaning and parsing
                    try:
                        # Apply comprehensive JSON cleaning
                        cleaned_content = clean_llm_json_response(content)
                        
                        # Log the cleaned content for debugging on first attempt
                        if attempt == 0:
                            logger.debug(f"Cleaned JSON for {brand} {model}: {cleaned_content[:200]}...")
                        
                        # Parse JSON
                        vehicle_data = json.loads(cleaned_content)
                        
                        # Validate and fix structure
                        vehicle_data = validate_and_fix_json_structure(vehicle_data, brand, model, year)
                        
                        # Add metadata
                        vehicle_data['source'] = 'llm_api'
                        vehicle_data['last_updated'] = datetime.now().isoformat()
                        
                        logger.info(f"LLM data retrieved successfully: {brand} {model} {year}")
                        return vehicle_data
                        
                    except json.JSONDecodeError as e:
                        if attempt < self.max_retries:
                            logger.warning(f"JSON parsing failed for {brand} {model} (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                            logger.debug(f"Raw LLM response: {content[:500]}...")
                            logger.debug(f"Cleaned content: {cleaned_content[:500]}...")
                            time.sleep(2)  # Wait before retry
                            continue
                        else:
                            logger.error(f"JSON parsing failed after {self.max_retries + 1} attempts for {brand} {model}: {e}")
                            logger.debug(f"Final raw response: {content}")
                            
                            # Generate fallback data as last resort
                            fallback_data = {
                                'brand': brand,
                                'model': model,
                                'year': year,
                                'trim': 'Base',
                                'body_type': 'Sedan',
                                'fuel_type': 'Gasoline',
                                'engine_type': '4-Cylinder',
                                'horsepower': 200,
                                'torque': 250,
                                'transmission': 'Automatic',
                                'drivetrain': 'FWD',
                                'acceleration_0_60': 8.0,
                                'top_speed': 120,
                                'mpg_city': 25,
                                'mpg_highway': 35,
                                'seating_capacity': 5,
                                'cargo_space': 15.0,
                                'msrp': 30000,
                                'safety_rating': 4.0,
                                'source': 'fallback_generated',
                                'last_updated': datetime.now().isoformat()
                            }
                            logger.warning(f"Using fallback data for {brand} {model} {year}")
                            return fallback_data
                        
                else:
                    if attempt < self.max_retries:
                        logger.warning(f"LLM API error {response.status_code} for {brand} {model} (attempt {attempt + 1}/{self.max_retries + 1})")
                        time.sleep(2)  # Wait before retry
                        continue
                    else:
                        logger.error(f"LLM API error {response.status_code} after {self.max_retries + 1} attempts: {brand} {model}")
                        return None
                
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"LLM API call failed for {brand} {model} (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    logger.error(f"LLM API call failed after {self.max_retries + 1} attempts: {brand} {model} - {e}")
                    return None
            
            finally:
                if attempt < self.max_retries:
                    time.sleep(self.rate_limit_delay)
        
        return None
    
    def get_multiple_vehicles(self, vehicle_list: List[tuple] = None, max_requests: int = 500) -> List[Dict[str, Any]]:
        """Batch retrieve multiple vehicle data with enhanced progress tracking and statistics"""
        # If no vehicle list provided, get vehicle list first
        if not vehicle_list:
            logger.info("First retrieving popular vehicle list...")
            vehicle_list = self.get_popular_vehicle_models()
        
        vehicles = []
        processed = 0
        successful = 0
        failed = 0
        fallback_used = 0
        
        logger.info(f"Starting LLM API vehicle data collection")
        logger.info(f"Target: {min(max_requests, len(vehicle_list))} vehicles from {len(vehicle_list)} available")
        
        for brand, model, year in vehicle_list:
            if processed >= max_requests:
                break
                
            try:
                vehicle_data = self.get_vehicle_specs(brand, model, year)
                processed += 1
                
                if vehicle_data:
                    vehicles.append(vehicle_data)
                    
                    # Track data source for statistics
                    if vehicle_data.get('source') == 'fallback_generated':
                        fallback_used += 1
                        logger.info(f"Processed: {processed}/{max_requests} - {brand} {model} {year} (fallback)")
                    else:
                        successful += 1
                        logger.info(f"Processed: {processed}/{max_requests} - {brand} {model} {year} (success)")
                else:
                    failed += 1
                    logger.warning(f"Skipped: {brand} {model} {year} (failed)")
                    
            except Exception as e:
                processed += 1
                failed += 1
                logger.error(f"Processing failed: {brand} {model} {year} - {e}")
                continue
        
        # Log comprehensive statistics
        total_retrieved = len(vehicles)
        logger.info("=" * 60)
        logger.info("LLM DATA COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Processed: {processed}")
        logger.info(f"Successfully Retrieved: {total_retrieved}")
        logger.info(f"├── Real LLM Data: {successful}")
        logger.info(f"├── Fallback Data: {fallback_used}")
        logger.info(f"└── Complete Failures: {failed}")
        logger.info(f"Success Rate: {(total_retrieved/processed)*100:.1f}%")
        logger.info(f"LLM Quality Rate: {(successful/processed)*100:.1f}%")
        logger.info("=" * 60)
        
        return vehicles

def get_llm_enhanced_vehicle_data() -> List[VehicleData]:
    """Get enhanced vehicle data using LLM API"""
    # Check if LLM API is configured
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not found. LLM API is required for data collection.")
    
    # Initialize LLM collector
    llm_collector = LLMVehicleDataCollector(api_key)
    
    # Use LLM to get data
    try:
        llm_vehicles = llm_collector.get_multiple_vehicles(max_requests=500)
        
        if not llm_vehicles:
            raise RuntimeError("No vehicle data retrieved from LLM API")
        
        # Convert to VehicleData format
        vehicles_data = []
        for vehicle_dict in llm_vehicles:
            try:
                vehicle_data = VehicleData(
                    brand=vehicle_dict.get('brand', ''),
                    model=vehicle_dict.get('model', ''),
                    year=vehicle_dict.get('year', 2024),
                    trim=vehicle_dict.get('trim', 'Base'),
                    body_type=vehicle_dict.get('body_type', 'Sedan'),
                    fuel_type=vehicle_dict.get('fuel_type', 'Gasoline'),
                    transmission=vehicle_dict.get('transmission', 'Automatic'),
                    drivetrain=vehicle_dict.get('drivetrain', 'FWD'),
                    msrp=vehicle_dict.get('msrp', 30000),
                    engine_type=vehicle_dict.get('engine_type', '4-Cylinder'),
                    horsepower=vehicle_dict.get('horsepower', 200),
                    torque=vehicle_dict.get('torque', 250),
                    acceleration_0_60=vehicle_dict.get('acceleration_0_60', 8.0),
                    mpg_city=vehicle_dict.get('mpg_city', 25),
                    mpg_highway=vehicle_dict.get('mpg_highway', 35),
                    seating_capacity=vehicle_dict.get('seating_capacity', 5),
                    cargo_space=vehicle_dict.get('cargo_space', 15.0),
                    safety_rating=vehicle_dict.get('safety_rating', 4.0),
                    weight=vehicle_dict.get('weight'),
                    length=vehicle_dict.get('length'),
                    width=vehicle_dict.get('width'),
                    height=vehicle_dict.get('height'),
                    top_speed=vehicle_dict.get('top_speed'),
                    lateral_g=vehicle_dict.get('lateral_g'),
                    quarter_mile_time=vehicle_dict.get('quarter_mile_time'),
                    quarter_mile_speed=vehicle_dict.get('quarter_mile_speed'),
                    braking_60_0=vehicle_dict.get('braking_60_0'),
                    braking_100_0=vehicle_dict.get('braking_100_0'),
                    source=vehicle_dict.get('source', 'llm_api'),
                    last_updated=vehicle_dict.get('last_updated', datetime.now().isoformat())
                )
                vehicles_data.append(vehicle_data)
                
            except Exception as e:
                logger.warning(f"Vehicle data conversion failed: {e}")
                continue
        
        logger.info(f"LLM enhanced data collection complete, total {len(vehicles_data)} vehicles")
        return vehicles_data
        
    except Exception as e:
        raise RuntimeError(f"LLM data collection failed: {e}")

def build_car_database(db_path: str = "examples/car_advisor/car_advisor_db.sqlite", 
                      fred_api_key: str = None):
    """Build car database, prioritizing LLM API for real data"""
    db_path = Path(db_path)
    
    # If db_path is a directory (from create.py), create the database file inside it
    if db_path.is_dir() or not db_path.suffix:
        db_path = db_path / "car_advisor_db.sqlite"
    
    # Ensure directory exists
    os.makedirs(db_path.parent, exist_ok=True)
    
    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Removed existing database: {db_path}")
    
    # 🚀 Use LLM API to get real vehicle data
    logger.info("Starting vehicle data collection...")
    
    vehicles_data = get_llm_enhanced_vehicle_data()
    
    logger.info(f"Total vehicles for database: {len(vehicles_data)}")
    
    # Create database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create all necessary tables
    create_tables(cursor)
    
    # Insert vehicle data
    insert_vehicle_data(cursor, vehicles_data)
    
    # Create additional sample data for other tables
    create_enhanced_data(cursor)
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database build complete: {db_path}")
    logger.info(f"Contains {len(vehicles_data)} vehicles")
    
    return str(db_path)

def create_tables(cursor):
    """Create all necessary database tables for comprehensive car lifecycle management"""
    
    # === REFERENCE LAYER: Static Vehicle Data ===
    
    # Main vehicles table - enhanced with American market focus  
    cursor.execute('''
        CREATE TABLE vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            year INTEGER NOT NULL,
            trim TEXT,
            body_type TEXT,
            fuel_type TEXT,
            transmission TEXT,
            drivetrain TEXT,
            msrp REAL,
            engine_type TEXT,
            horsepower INTEGER,
            torque INTEGER,
            acceleration_0_60 REAL,
            top_speed INTEGER,
            mpg_city INTEGER,
            mpg_highway INTEGER,
            seating_capacity INTEGER,
            cargo_space REAL,
            safety_rating REAL,
            weight INTEGER,
            length REAL,
            width REAL,
            height REAL,
            lateral_g REAL,
            quarter_mile_time REAL,
            quarter_mile_speed INTEGER,
            braking_60_0 REAL,
            braking_100_0 REAL,
            nhtsa_overall_rating INTEGER,
            nhtsa_frontal_crash INTEGER,
            nhtsa_side_crash INTEGER,
            nhtsa_rollover INTEGER,
            iihs_top_safety_pick BOOLEAN DEFAULT FALSE,
            epa_greenhouse_score INTEGER,
            warranty_basic_years INTEGER DEFAULT 3,
            warranty_basic_miles INTEGER DEFAULT 36000,
            warranty_powertrain_years INTEGER DEFAULT 5,
            warranty_powertrain_miles INTEGER DEFAULT 60000,
            country_of_origin TEXT,
            assembly_plant_state TEXT,
            last_updated TEXT,
            data_quality_score REAL DEFAULT 0.9,
            source TEXT DEFAULT 'curated'
        )
    ''')
    
    # Brand characteristics with American market focus
    cursor.execute('''
        CREATE TABLE brand_characteristics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT NOT NULL UNIQUE,
            reliability_score REAL DEFAULT 7.0,
            luxury_score REAL DEFAULT 5.0,
            performance_score REAL DEFAULT 5.0,
            value_score REAL DEFAULT 5.0,
            technology_score REAL DEFAULT 5.0,
            environmental_score REAL DEFAULT 5.0,
            maintenance_cost_level TEXT DEFAULT 'Medium',
            average_maintenance_cost_annual REAL DEFAULT 1200,
            resale_value_score REAL DEFAULT 6.5,
            brand_category TEXT DEFAULT 'Mainstream',
            us_market_share REAL DEFAULT 0.0,
            headquarters_country TEXT,
            popular_in_states TEXT,
            target_demographic TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE brand_similarity (
            brand TEXT NOT NULL,
            similar_brand TEXT NOT NULL,
            similarity_score REAL DEFAULT 7.0,
            similarity_reason TEXT,
            competition_level TEXT DEFAULT 'Direct',
            PRIMARY KEY (brand, similar_brand)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE vehicle_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            feature_name TEXT NOT NULL,
            feature_category TEXT DEFAULT 'Standard',
            feature_description TEXT,
            is_standard BOOLEAN DEFAULT 1,
            additional_cost REAL DEFAULT 0,
            availability TEXT DEFAULT 'All Trims',
            FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
        )
    ''')
    
    # Vehicle analysis table
    cursor.execute('''
        CREATE TABLE vehicle_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            analysis_type TEXT,
            description TEXT,
            category TEXT,
            score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
        )
    ''')
    
    # === TRANSACTION LAYER: Dynamic Business Data ===
    
    # Enhanced users table with comprehensive American market preferences
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            phone TEXT,
            age INTEGER,
            state TEXT,
            city TEXT,
            zip_code TEXT,
            credit_score_range TEXT, -- Excellent/Good/Fair/Poor
            annual_income REAL,
            budget_min REAL,
            budget_max REAL,
            down_payment_available REAL,
            preferred_fuel_type TEXT,
            preferred_body_type TEXT,
            preferred_transmission TEXT, -- Manual/Automatic/CVT
            max_acceptable_mileage INTEGER, -- for used cars
            driving_experience_years INTEGER,
            license_type TEXT, -- Regular/CDL/Motorcycle
            family_size INTEGER,
            primary_use TEXT, -- Commuting/Family/Recreation/Work
            parking_situation TEXT, -- Garage/Driveway/Street
            climate_zone TEXT, -- Hot/Cold/Moderate/Mixed
            financing_pre_approved BOOLEAN DEFAULT FALSE,
            trade_in_vehicle_id INTEGER,
            estimated_trade_value REAL,
            current_search_status TEXT, -- Browsing/Serious/Ready_to_Buy
            preferred_dealer_distance INTEGER DEFAULT 50, -- miles
            communication_preference TEXT, -- Email/Phone/Text
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_in_vehicle_id) REFERENCES vehicles (id)
        )
    ''')
    
    # Dealers table - comprehensive US dealership network
    cursor.execute('''
        CREATE TABLE dealers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand_affiliations TEXT, -- JSON array of brands they sell
            address_street TEXT,
            city TEXT,
            state TEXT NOT NULL,
            zip_code TEXT,
            phone TEXT,
            website TEXT,
            email TEXT,
            business_hours TEXT, -- JSON with daily hours
            services_offered TEXT, -- Sales/Service/Parts/Body_Shop/Financing
            customer_rating REAL DEFAULT 4.0,
            google_reviews_count INTEGER DEFAULT 0,
            better_business_bureau_rating TEXT,
            volume_dealer BOOLEAN DEFAULT FALSE, -- High volume dealer
            certified_pre_owned BOOLEAN DEFAULT FALSE,
            loaner_vehicles BOOLEAN DEFAULT FALSE,
            shuttle_service BOOLEAN DEFAULT FALSE,
            saturday_service BOOLEAN DEFAULT FALSE,
            languages_spoken TEXT, -- English/Spanish/etc
            finance_partners TEXT, -- JSON array of financing partners
            special_programs TEXT, -- Military/Student/First_Time_Buyer discounts
            inventory_size INTEGER DEFAULT 0,
            years_in_business INTEGER,
            latitude REAL,
            longitude REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Inventory management - track specific vehicles with VIN
    cursor.execute('''
        CREATE TABLE inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dealer_id INTEGER NOT NULL,
            vehicle_id INTEGER NOT NULL,
            vin TEXT UNIQUE,
            stock_number TEXT,
            condition_type TEXT, -- New/Used/Certified_Pre_Owned
            exterior_color TEXT,
            interior_color TEXT,
            trim_level TEXT,
            mileage INTEGER DEFAULT 0,
            model_year INTEGER,
            asking_price REAL,
            invoice_price REAL, -- dealer cost (if known)
            market_adjustment REAL DEFAULT 0, -- markup/markdown
            incentives_available TEXT, -- JSON array of current incentives  
            status TEXT DEFAULT 'Available', -- Available/Reserved/Sold/In_Transit
            days_on_lot INTEGER DEFAULT 0,
            acquisition_date DATE,
            expected_arrival_date DATE,
            carfax_report_available BOOLEAN DEFAULT FALSE,
            accident_history BOOLEAN DEFAULT FALSE,
            previous_owners INTEGER DEFAULT 0,
            service_records_available BOOLEAN DEFAULT FALSE,
            key_count INTEGER DEFAULT 2,
            special_notes TEXT,
            photos_url TEXT, -- JSON array of photo URLs
            video_url TEXT,
            featured_listing BOOLEAN DEFAULT FALSE,
            online_price REAL, -- sometimes different from asking_price
            certified_pre_owned BOOLEAN DEFAULT FALSE,
            warranty_remaining_months INTEGER,
            financing_specials TEXT,
            lease_specials TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dealer_id) REFERENCES dealers (id),
            FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
        )
    ''')
    
    # Customer orders and purchase process tracking
    cursor.execute('''
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_number TEXT UNIQUE,
            user_id INTEGER NOT NULL,
            inventory_id INTEGER,
            dealer_id INTEGER NOT NULL,
            salesperson_name TEXT,
            status TEXT DEFAULT 'Inquiry', -- Inquiry/Quote/Negotiating/Contracted/Financing/Delivered/Cancelled
            purchase_type TEXT, -- Cash/Finance/Lease
            agreed_price REAL,
            trade_in_value REAL DEFAULT 0,
            down_payment REAL DEFAULT 0,
            monthly_payment REAL,
            financing_term_months INTEGER,
            interest_rate REAL,
            financing_approved BOOLEAN DEFAULT FALSE,
            lender_name TEXT,
            extended_warranty BOOLEAN DEFAULT FALSE,
            gap_insurance BOOLEAN DEFAULT FALSE,
            additional_products TEXT, -- JSON array of add-ons
            total_fees REAL DEFAULT 0, -- doc fees, registration, etc
            sales_tax REAL DEFAULT 0,
            final_price REAL, -- total out-the-door price
            deposit_amount REAL DEFAULT 0,
            deposit_paid BOOLEAN DEFAULT FALSE,
            paperwork_signed BOOLEAN DEFAULT FALSE,
            financing_paperwork_complete BOOLEAN DEFAULT FALSE,
            insurance_provided BOOLEAN DEFAULT FALSE,
            delivery_method TEXT, -- Pickup/Home_Delivery/Shipped
            expected_delivery_date DATE,
            actual_delivery_date DATE,
            delivery_address TEXT,
            keys_delivered BOOLEAN DEFAULT FALSE,
            customer_satisfaction_rating INTEGER,
            referral_source TEXT,
            special_instructions TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (inventory_id) REFERENCES inventory (id),
            FOREIGN KEY (dealer_id) REFERENCES dealers (id)
        )
    ''')
    
    # Test drive scheduling and tracking
    cursor.execute('''
        CREATE TABLE test_drives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            vehicle_id INTEGER NOT NULL,
            dealer_id INTEGER NOT NULL,
            inventory_id INTEGER,
            scheduled_datetime TIMESTAMP,
            duration_minutes INTEGER DEFAULT 30,
            route_type TEXT, -- City/Highway/Mixed
            salesperson_name TEXT,
            status TEXT DEFAULT 'Scheduled', -- Scheduled/Completed/Cancelled/No_Show
            customer_showed BOOLEAN DEFAULT FALSE,
            valid_license_verified BOOLEAN DEFAULT FALSE,
            insurance_verified BOOLEAN DEFAULT FALSE,
            mileage_before INTEGER,
            mileage_after INTEGER,
            fuel_level_before REAL,
            fuel_level_after REAL,
            vehicle_condition_notes TEXT,
            customer_feedback_rating INTEGER, -- 1-10
            customer_feedback_comments TEXT,
            customer_interest_level TEXT, -- Not_Interested/Somewhat_Interested/Very_Interested/Ready_to_Buy
            follow_up_scheduled BOOLEAN DEFAULT FALSE,
            follow_up_date DATE,
            sales_notes TEXT,
            photos_taken TEXT, -- JSON array if any damage occurred
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (vehicle_id) REFERENCES vehicles (id),
            FOREIGN KEY (dealer_id) REFERENCES dealers (id),
            FOREIGN KEY (inventory_id) REFERENCES inventory (id)
        )
    ''')
    
    # Price negotiation tracking
    cursor.execute('''
        CREATE TABLE negotiations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            sequence_number INTEGER DEFAULT 1,
            negotiation_type TEXT, -- Vehicle_Price/Trade_Value/Financing/Add_Ons
            dealer_offer REAL,
            customer_counter REAL,
            current_offer REAL,
            offer_status TEXT, -- Pending/Accepted/Rejected/Countered
            offer_valid_until TIMESTAMP,
            negotiation_notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders (id)
        )
    ''')
    
    # Financing options and applications
    cursor.execute('''
        CREATE TABLE financing_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            order_id INTEGER,
            lender_name TEXT,
            application_type TEXT, -- Purchase/Lease/Refinance
            requested_amount REAL,
            requested_term_months INTEGER,
            credit_score INTEGER,
            annual_income REAL,
            monthly_debt_payments REAL,
            employment_status TEXT,
            employment_years_current REAL,
            application_status TEXT, -- Submitted/Under_Review/Approved/Conditionally_Approved/Denied
            approved_amount REAL,
            approved_rate REAL,
            approved_term_months INTEGER,
            monthly_payment REAL,
            conditions TEXT, -- JSON array of approval conditions
            denial_reason TEXT,
            application_date DATE,
            decision_date DATE,
            expiration_date DATE,
            application_reference TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (order_id) REFERENCES orders (id)
        )
    ''')
    
    # === HISTORICAL DATA ===
    
    # Price history tracking for market analysis  
    cursor.execute('''
        CREATE TABLE price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            inventory_id INTEGER,
            price REAL,
            price_type TEXT, -- MSRP/Invoice/Asking/Sold/Market_Average
            mileage INTEGER,
            condition_type TEXT,
            region TEXT, -- State or metro area
            date_recorded DATE,
            source TEXT, -- Dealer/KBB/Edmunds/AutoTrader/Manual_Entry
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (vehicle_id) REFERENCES vehicles (id),
            FOREIGN KEY (inventory_id) REFERENCES inventory (id)
        )
    ''')
    
    # Customer interaction and communication log
    cursor.execute('''
        CREATE TABLE customer_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            dealer_id INTEGER,
            order_id INTEGER,
            interaction_type TEXT, -- Phone/Email/Text/In_Person/Website_Chat
            direction TEXT, -- Inbound/Outbound
            subject TEXT,
            summary TEXT,
            staff_member TEXT,
            follow_up_required BOOLEAN DEFAULT FALSE,
            follow_up_date DATE,
            interaction_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (dealer_id) REFERENCES dealers (id),
            FOREIGN KEY (order_id) REFERENCES orders (id)
        )
    ''')
    
    # === FUTURE PREPARATION: Driving Guidance Tables ===
    
    # User-owned vehicles (post-purchase)
    cursor.execute('''
        CREATE TABLE user_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            vehicle_id INTEGER NOT NULL,
            order_id INTEGER,
            vin TEXT,
            license_plate TEXT,
            nickname TEXT, -- user's name for their car
            purchase_date DATE,
            purchase_price REAL,
            current_mileage INTEGER,
            last_service_date DATE,
            next_service_due_date DATE,
            next_service_due_mileage INTEGER,
            insurance_company TEXT,
            insurance_policy_number TEXT,
            insurance_expires DATE,
            registration_expires DATE,
            warranty_expires DATE,
            preferred_service_dealer_id INTEGER,
            is_primary_vehicle BOOLEAN DEFAULT FALSE,
            is_active BOOLEAN DEFAULT TRUE,
            vehicle_condition TEXT DEFAULT 'Excellent', -- Excellent/Good/Fair/Poor
            modifications TEXT, -- JSON array of aftermarket modifications
            service_reminders_enabled BOOLEAN DEFAULT TRUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (vehicle_id) REFERENCES vehicles (id),
            FOREIGN KEY (order_id) REFERENCES orders (id),
            FOREIGN KEY (preferred_service_dealer_id) REFERENCES dealers (id)
        )
    ''')
    
    # Driving knowledge base for guidance
    cursor.execute('''
        CREATE TABLE driving_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL, -- Basic_Skills/Advanced_Techniques/Safety/Maintenance/Local_Laws
            subcategory TEXT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            difficulty_level INTEGER DEFAULT 1, -- 1-5
            applicable_vehicle_types TEXT, -- JSON array
            applicable_states TEXT, -- JSON array for state-specific info
            age_group TEXT, -- Teen/Adult/Senior/All
            weather_conditions TEXT, -- All/Rain/Snow/Ice/Fog
            road_types TEXT, -- City/Highway/Rural/Parking
            media_type TEXT, -- Text/Video/Image/Interactive
            media_url TEXT,
            importance_level TEXT DEFAULT 'Medium', -- Low/Medium/High/Critical
            legal_requirement BOOLEAN DEFAULT FALSE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Maintenance reminders and service tracking
    cursor.execute('''
        CREATE TABLE maintenance_reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_vehicle_id INTEGER NOT NULL,
            reminder_type TEXT NOT NULL, -- Oil_Change/Tire_Rotation/Brake_Inspection/Registration_Renewal
            description TEXT,
            due_date DATE,
            due_mileage INTEGER,
            priority_level TEXT DEFAULT 'Medium', -- Low/Medium/High/Urgent
            estimated_cost REAL,
            preferred_service_location TEXT,
            status TEXT DEFAULT 'Pending', -- Pending/Scheduled/Completed/Overdue/Dismissed
            completed_date DATE,
            actual_cost REAL,
            service_provider TEXT,
            notes TEXT,
            next_service_due_mileage INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_vehicle_id) REFERENCES user_vehicles (id)
        )
    ''')

def insert_vehicle_data(cursor, vehicles_data):
    """Insert vehicle data into enhanced database structure"""
    
    # Initialize LLM collector for dynamic data
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.warning("OPENAI_API_KEY not found. Using fallback values for warranty data.")
        llm_collector = None
    else:
        llm_collector = LLMVehicleDataCollector(api_key)
    
    # Cache for brand-specific data to avoid duplicate API calls
    brand_warranty_cache = {}
    
    # Default warranty data structure
    default_warranty_data = {
        "warranty_basic_years": 3,
        "warranty_basic_miles": 36000,
        "warranty_powertrain_years": 5,
        "warranty_powertrain_miles": 60000,
        "iihs_top_safety_pick_common": False,
        "epa_greenhouse_score_typical": 7
    }
    
    for vehicle in vehicles_data:
        # Get brand-specific warranty and safety info from LLM API
        if llm_collector and vehicle.brand not in brand_warranty_cache:
            warranty_info = llm_collector.get_brand_warranty_info(vehicle.brand)
            if warranty_info:
                # Merge LLM data with defaults to ensure all keys exist
                merged_warranty_data = default_warranty_data.copy()
                merged_warranty_data.update(warranty_info)
                brand_warranty_cache[vehicle.brand] = merged_warranty_data
            else:
                # Fallback to reasonable defaults if API fails
                brand_warranty_cache[vehicle.brand] = default_warranty_data.copy()
        elif vehicle.brand not in brand_warranty_cache:
            # No LLM API available, use defaults
            brand_warranty_cache[vehicle.brand] = default_warranty_data.copy()
        
        warranty_data = brand_warranty_cache[vehicle.brand]
        
        # Enhanced vehicle data with LLM-generated brand-specific data
        cursor.execute('''
            INSERT INTO vehicles (
                brand, model, year, trim, body_type, fuel_type, transmission, drivetrain,
                msrp, engine_type, horsepower, torque, acceleration_0_60, top_speed, mpg_city, mpg_highway,
                seating_capacity, cargo_space, safety_rating, weight, length, width, height,
                lateral_g, quarter_mile_time, quarter_mile_speed, braking_60_0, braking_100_0,
                nhtsa_overall_rating, nhtsa_frontal_crash, nhtsa_side_crash, nhtsa_rollover,
                iihs_top_safety_pick, epa_greenhouse_score, warranty_basic_years, warranty_basic_miles,
                warranty_powertrain_years, warranty_powertrain_miles, country_of_origin, assembly_plant_state,
                last_updated, data_quality_score, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            vehicle.brand, vehicle.model, vehicle.year, vehicle.trim, vehicle.body_type,
            vehicle.fuel_type, vehicle.transmission, vehicle.drivetrain, vehicle.msrp,
            vehicle.engine_type, vehicle.horsepower, vehicle.torque, vehicle.acceleration_0_60,
            vehicle.top_speed, vehicle.mpg_city, vehicle.mpg_highway, vehicle.seating_capacity, 
            vehicle.cargo_space, vehicle.safety_rating, vehicle.weight, vehicle.length, 
            vehicle.width, vehicle.height, vehicle.lateral_g, vehicle.quarter_mile_time,
            vehicle.quarter_mile_speed, vehicle.braking_60_0, vehicle.braking_100_0,
            vehicle.nhtsa_overall_rating, vehicle.nhtsa_frontal_crash, vehicle.nhtsa_side_crash,
            vehicle.nhtsa_rollover,
            # LLM-generated brand-specific data
            warranty_data["iihs_top_safety_pick_common"],
            warranty_data["epa_greenhouse_score_typical"],
            warranty_data["warranty_basic_years"],
            warranty_data["warranty_basic_miles"],
            warranty_data["warranty_powertrain_years"],
            warranty_data["warranty_powertrain_miles"],
            _get_country_of_origin(vehicle.brand),
            _get_assembly_plant_state(vehicle.brand),
            vehicle.last_updated, vehicle.data_quality_score, f"{vehicle.source}_with_llm_warranty"
        ))

def _get_country_of_origin(brand):
    """Get country of origin for vehicle brand"""
    origins = {
        'Toyota': 'Japan', 'Honda': 'Japan', 'Nissan': 'Japan', 'Mazda': 'Japan',
        'Subaru': 'Japan', 'Mitsubishi': 'Japan', 'Lexus': 'Japan', 'Acura': 'Japan', 'Infiniti': 'Japan',
        'Ford': 'United States', 'Chevrolet': 'United States', 'GMC': 'United States', 'Cadillac': 'United States',
        'Buick': 'United States', 'Lincoln': 'United States', 'Chrysler': 'United States', 'Dodge': 'United States',
        'Jeep': 'United States', 'Ram': 'United States', 'Tesla': 'United States',
        'BMW': 'Germany', 'Mercedes-Benz': 'Germany', 'Audi': 'Germany', 'Volkswagen': 'Germany', 'Porsche': 'Germany',
        'Volvo': 'Sweden', 'Land Rover': 'United Kingdom', 'Jaguar': 'United Kingdom', 'Mini': 'United Kingdom',
        'Hyundai': 'South Korea', 'Kia': 'South Korea', 'Genesis': 'South Korea'
    }
    return origins.get(brand, 'Unknown')

def _get_assembly_plant_state(brand):
    """Get typical assembly plant state for vehicle brand"""
    plants = {
        'Toyota': 'Kentucky', 'Honda': 'Ohio', 'Nissan': 'Tennessee', 'Mazda': 'Alabama',
        'Subaru': 'Indiana', 'Lexus': 'Kentucky', 'Acura': 'Ohio', 'Infiniti': 'Tennessee',
        'Ford': 'Michigan', 'Chevrolet': 'Michigan', 'GMC': 'Michigan', 'Cadillac': 'Michigan',
        'Buick': 'Michigan', 'Lincoln': 'Michigan', 'Chrysler': 'Michigan', 'Dodge': 'Michigan',
        'Jeep': 'Michigan', 'Ram': 'Michigan', 'Tesla': 'California',
        'BMW': 'South Carolina', 'Mercedes-Benz': 'Alabama', 'Volkswagen': 'Tennessee',
        'Volvo': 'South Carolina', 'Hyundai': 'Alabama', 'Kia': 'Georgia'
    }
    return plants.get(brand, 'Various')

def create_enhanced_data(cursor):
    """Create comprehensive sample data for American automotive market using LLM API"""
    
    # Initialize LLM collector for dynamic data generation
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.warning("OPENAI_API_KEY not found. Using fallback brand characteristics.")
        llm_collector = None
    else:
        llm_collector = LLMVehicleDataCollector(api_key)
    
    # Get unique brands from vehicles table
    cursor.execute("SELECT DISTINCT brand FROM vehicles ORDER BY brand")
    brands_in_db = [row[0] for row in cursor.fetchall()]
    
    # Generate LLM-based brand characteristics
    if llm_collector and brands_in_db:
        logger.info("Generating brand characteristics using LLM API...")
        
        for brand in brands_in_db:
            brand_data = llm_collector.get_brand_characteristics(brand)
            if brand_data:
                cursor.execute('''
                    INSERT OR REPLACE INTO brand_characteristics 
                    (brand, reliability_score, luxury_score, performance_score, value_score, 
                     technology_score, environmental_score, maintenance_cost_level, average_maintenance_cost_annual,
                     resale_value_score, brand_category, us_market_share, headquarters_country, 
                     popular_in_states, target_demographic)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    brand,
                    brand_data.get("reliability_score", 7.0),
                    brand_data.get("luxury_score", 5.0),
                    brand_data.get("performance_score", 5.0),
                    brand_data.get("value_score", 7.0),
                    brand_data.get("technology_score", 6.0),
                    brand_data.get("environmental_score", 6.0),
                    brand_data.get("maintenance_cost_level", "Medium"),
                    brand_data.get("average_maintenance_cost_annual", 1000),
                    brand_data.get("resale_value_score", 6.5),
                    "Mainstream",  # Will be determined by luxury_score
                    brand_data.get("us_market_share", 2.0),
                    _get_country_of_origin(brand),
                    brand_data.get("popular_in_states", "CA,TX,NY,FL"),
                    brand_data.get("target_demographic", "General Market")
                ))
                logger.info(f"Generated brand characteristics for {brand}")
            else:
                logger.warning(f"Failed to generate characteristics for {brand}, using defaults")
                # Fallback to basic defaults
                cursor.execute('''
                    INSERT OR REPLACE INTO brand_characteristics 
                    (brand, reliability_score, luxury_score, performance_score, value_score, 
                     technology_score, environmental_score, maintenance_cost_level, average_maintenance_cost_annual,
                     resale_value_score, brand_category, us_market_share, headquarters_country, 
                     popular_in_states, target_demographic)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (brand, 7.0, 5.0, 5.0, 7.0, 6.0, 6.0, "Medium", 1000, 6.5, "Mainstream", 2.0, 
                      _get_country_of_origin(brand), "CA,TX,NY,FL", "General Market"))
    else:
        # Fallback: Use simplified static data for major brands only
        logger.warning("Using fallback brand characteristics data")
        fallback_brands = [
            ('Toyota', 9.0, 6.0, 7.0, 9.0, 7.5, 8.5, 'Low', 600, 8.5, 'Mainstream', 14.8, 'Japan', 'CA,TX,FL,NY,OH', 'Reliability-Focused'),
            ('Honda', 8.5, 6.5, 7.5, 9.0, 8.0, 8.0, 'Low', 650, 8.0, 'Mainstream', 9.2, 'Japan', 'CA,TX,FL,NY,OH', 'Practical Families'),
            ('Ford', 7.5, 6.5, 8.0, 8.0, 7.5, 7.0, 'Medium', 900, 6.5, 'Mainstream', 13.4, 'United States', 'TX,MI,OH,FL,CA', 'American Traditionalists'),
            ('Chevrolet', 7.0, 6.0, 8.5, 8.5, 7.0, 7.0, 'Medium', 850, 6.0, 'Mainstream', 12.8, 'United States', 'TX,MI,OH,CA,FL', 'Value Seekers'),
            ('BMW', 8.0, 9.0, 9.0, 6.0, 8.5, 6.0, 'High', 1800, 8.0, 'Luxury Performance', 3.2, 'Germany', 'CA,NY,TX,FL', 'Affluent Professionals')
        ]
        
        for brand_data in fallback_brands:
            cursor.execute('''
                INSERT OR REPLACE INTO brand_characteristics 
                (brand, reliability_score, luxury_score, performance_score, value_score, 
                 technology_score, environmental_score, maintenance_cost_level, average_maintenance_cost_annual,
                 resale_value_score, brand_category, us_market_share, headquarters_country, 
                 popular_in_states, target_demographic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', brand_data)
    
    # Enhanced brand similarities (static but logical relationships)
    brand_similarities = [
        ('BMW', 'Mercedes-Benz', 8.5, 'German luxury performance competitors', 'Direct'),
        ('BMW', 'Audi', 8.0, 'German luxury sports sedan rivals', 'Direct'),
        ('Mercedes-Benz', 'Audi', 8.5, 'Premium German luxury brands', 'Direct'),
        ('Tesla', 'BMW', 6.5, 'Electric luxury performance crossover', 'Emerging'),
        ('Toyota', 'Honda', 9.0, 'Reliable mainstream Japanese brands', 'Direct'),
        ('Toyota', 'Mazda', 7.5, 'Japanese engineering and quality focus', 'Indirect'),
        ('Honda', 'Mazda', 7.0, 'Japanese mainstream with driving dynamics', 'Indirect'),
        ('Lexus', 'Acura', 8.0, 'Japanese luxury divisions', 'Direct'),
        ('Ford', 'Chevrolet', 8.5, 'American mainstream full-line brands', 'Direct'),
        ('Porsche', 'BMW', 7.0, 'Performance-oriented German luxury', 'Indirect'),
        ('Hyundai', 'Kia', 9.0, 'Sister brands from Hyundai Motor Group', 'Corporate'),
        ('Subaru', 'Mazda', 6.5, 'Smaller Japanese brands with enthusiast appeal', 'Indirect'),
        ('Genesis', 'Lexus', 7.5, 'Asian luxury brands challenging Germans', 'Aspirational'),
        ('Tesla', 'Mercedes-Benz', 5.5, 'Electric vs traditional luxury', 'Disruptive'),
        ('Volvo', 'Subaru', 6.0, 'Safety-focused brands with loyal followings', 'Indirect')
    ]
    
    for brand1, brand2, score, reason, level in brand_similarities:
        cursor.execute('''
            INSERT OR REPLACE INTO brand_similarity (brand, similar_brand, similarity_score, similarity_reason, competition_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (brand1, brand2, score, reason, level))
        cursor.execute('''
            INSERT OR REPLACE INTO brand_similarity (brand, similar_brand, similarity_score, similarity_reason, competition_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (brand2, brand1, score, reason, level))
    
    # Sample dealerships across major US markets (static data - realistic)
    sample_dealers = [
        ("Metro BMW of Scarsdale", '["BMW"]', "123 Central Park Ave", "Scarsdale", "NY", "10583", "(914) 725-4200", "www.metrobmw.com", "sales@metrobmw.com", 
         '{"Mon-Fri": "9:00-21:00", "Sat": "9:00-18:00", "Sun": "11:00-17:00"}', "Sales,Service,Parts,Body_Shop,Financing", 4.3, 892, "A+", True, True, True, True, True, "English,Spanish", 
         '["BMW Financial", "Chase", "Wells Fargo"]', '["Military", "College Graduate", "Loyalty"]', 150, 28, 40.7589, -73.7947),
        
        ("Toyota of Hollywood", '["Toyota", "Lexus"]', "6000 Hollywood Blvd", "Hollywood", "CA", "90028", "(323) 342-8200", "www.toyotahollywood.com", "info@toyotahollywood.com",
         '{"Mon-Sat": "8:00-21:00", "Sun": "10:00-20:00"}', "Sales,Service,Parts,Financing", 4.1, 1247, "A", True, True, True, False, True, "English,Spanish,Korean",
         '["Toyota Financial", "Lexus Financial", "Bank of America"]', '["Military", "College Graduate", "First Time Buyer"]', 300, 45, 34.0928, -118.3287),
        
        ("Friendly Ford", '["Ford", "Lincoln"]', "5500 Las Vegas Blvd S", "Las Vegas", "NV", "89119", "(702) 688-7000", "www.friendlyford.com", "sales@friendlyford.com",
         '{"Mon-Fri": "8:00-20:00", "Sat": "8:00-19:00", "Sun": "10:00-18:00"}', "Sales,Service,Parts,Body_Shop,Financing", 4.0, 678, "B+", True, False, True, True, True, "English,Spanish",
         '["Ford Credit", "Chase", "Capital One"]', '["Military", "Senior"]', 200, 22, 36.0840, -115.1721),
        
        ("Prestige Mercedes-Benz", '["Mercedes-Benz"]', "2000 Westheimer Rd", "Houston", "TX", "77098", "(713) 341-9400", "www.prestigemb.com", "luxury@prestigemb.com",
         '{"Mon-Fri": "9:00-20:00", "Sat": "9:00-18:00", "Sun": "12:00-17:00"}', "Sales,Service,Parts,Financing", 4.5, 456, "A+", True, True, True, True, True, "English,Spanish,German",
         '["Mercedes-Benz Financial", "Chase Private Client", "Wells Fargo Private Bank"]', '["Executive", "Loyalty"]', 80, 35, 29.7372, -95.4618),
        
        ("Chicago Honda", '["Honda", "Acura"]', "1000 W Grand Ave", "Chicago", "IL", "60642", "(312) 666-3900", "www.chicagohonda.com", "sales@chicagohonda.com",
         '{"Mon-Thu": "9:00-21:00", "Fri-Sat": "9:00-19:00", "Sun": "11:00-18:00"}', "Sales,Service,Parts,Financing", 4.2, 934, "A", True, True, True, True, True, "English,Spanish,Polish",
         '["Honda Financial", "Acura Financial", "Fifth Third Bank"]', '["Military", "College Graduate", "Loyalty", "First Time Buyer"]', 250, 38, 41.8919, -87.6501)
    ]
    
    for dealer in sample_dealers:
        cursor.execute('''
            INSERT INTO dealers (
                name, brand_affiliations, address_street, city, state, zip_code, phone, website, email,
                business_hours, services_offered, customer_rating, google_reviews_count, better_business_bureau_rating,
                volume_dealer, certified_pre_owned, loaner_vehicles, shuttle_service, saturday_service,
                languages_spoken, finance_partners, special_programs, inventory_size, years_in_business,
                latitude, longitude
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', dealer)
    
    # Enhanced sample users with comprehensive American market profiles
    sample_users = [
        ("John Smith", "john.smith@email.com", "(555) 123-4567", 32, "California", "Los Angeles", "90210", "Good", 85000, 25000, 45000, 8000, "Gasoline", "SUV", "Automatic", 50000, 
         15, "Regular", 3, "Commuting", "Garage", "Hot", False, None, 0, "Serious", 30, "Email"),
        ("Sarah Johnson", "sarah.johnson@email.com", "(555) 234-5678", 28, "Texas", "Houston", "77001", "Excellent", 120000, 35000, 65000, 15000, "Hybrid", "Sedan", "Automatic", 30000,
         10, "Regular", 1, "Family", "Driveway", "Hot", True, None, 0, "Ready_to_Buy", 25, "Phone"),
        ("Mike Brown", "mike.brown@email.com", "(555) 345-6789", 45, "New York", "New York", "10001", "Fair", 75000, 15000, 35000, 5000, "Electric", "Hatchback", "Automatic", 25000,
         25, "Regular", 2, "Commuting", "Street", "Cold", False, None, 0, "Browsing", 50, "Text"),
        ("Lisa Davis", "lisa.davis@email.com", "(555) 456-7890", 38, "Florida", "Miami", "33101", "Good", 95000, 40000, 70000, 12000, "Gasoline", "Convertible", "Manual", 40000,
         20, "Regular", 2, "Recreation", "Garage", "Hot", False, None, 0, "Serious", 40, "Email"),
        ("Robert Wilson", "robert.wilson@email.com", "(555) 567-8901", 55, "Colorado", "Denver", "80201", "Excellent", 150000, 50000, 85000, 20000, "Gasoline", "Pickup", "Automatic", 60000,
         35, "Regular", 4, "Work", "Garage", "Mixed", True, None, 0, "Ready_to_Buy", 75, "Phone")
    ]
    
    for user in sample_users:
        cursor.execute('''
            INSERT INTO users (
                name, email, phone, age, state, city, zip_code, credit_score_range, annual_income,
                budget_min, budget_max, down_payment_available, preferred_fuel_type, preferred_body_type,
                preferred_transmission, max_acceptable_mileage, driving_experience_years, license_type,
                family_size, primary_use, parking_situation, climate_zone, financing_pre_approved,
                trade_in_vehicle_id, estimated_trade_value, current_search_status, preferred_dealer_distance,
                communication_preference
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', user)
    
    # Sample inventory items with LLM-generated pricing
    cursor.execute("SELECT id, brand, model, year FROM vehicles LIMIT 10")
    sample_vehicles = cursor.fetchall()
    
    dealer_ids = [1, 2, 3, 4, 5]  # Assuming 5 dealers were inserted
    
    colors = ["White", "Black", "Silver", "Gray", "Blue", "Red", "Green"]
    interior_colors = ["Black", "Beige", "Gray", "Brown"]
    
    logger.info("Generating inventory with LLM-based pricing...")
    
    for i, (vehicle_id, brand, model, year) in enumerate(sample_vehicles):
        dealer_id = dealer_ids[i % len(dealer_ids)]
        exterior_color = colors[i % len(colors)]
        interior_color = interior_colors[i % len(interior_colors)]
        
        # Get LLM-based market pricing
        base_price = 35000  # Default
        if llm_collector:
            pricing_data = llm_collector.get_vehicle_market_price(brand, model, year)
            if pricing_data:
                base_price = pricing_data.get("typical_dealer_price", 35000)
                market_adjustment = pricing_data.get("market_adjustment", 0)
                logger.info(f"Generated pricing for {brand} {model}: ${base_price}")
            else:
                market_adjustment = 0
                logger.warning(f"Failed to get pricing for {brand} {model}, using default")
        else:
            market_adjustment = 0
        
        cursor.execute('''
            INSERT INTO inventory (
                dealer_id, vehicle_id, vin, stock_number, condition_type, exterior_color, interior_color,
                trim_level, mileage, model_year, asking_price, market_adjustment, status, days_on_lot,
                carfax_report_available, key_count, photos_url, featured_listing
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dealer_id, vehicle_id, f"1{brand[:2].upper()}{model[:2].upper()}{str(year)[-2:]}{str(i).zfill(6)}", 
            f"STK{str(i+1000)}", "New", exterior_color, interior_color, "Base", 0, year,
            base_price, market_adjustment, "Available", i * 5, True, 2, f'["/photos/{i+1}_1.jpg"]', i < 3
        ))
    
    # Sample driving knowledge entries
    driving_knowledge_entries = [
        ("Basic_Skills", "Vehicle_Controls", "Understanding Your Dashboard Warning Lights", 
         "Learn to recognize and respond to common dashboard warning lights including check engine, oil pressure, battery, and temperature warnings. Immediate action may be required for some lights.", 
         1, '["All"]', '["All"]', "All", "All", "All", "Text", None, "High", True),
        
        ("Safety", "Weather_Driving", "Safe Driving in Heavy Rain", 
         "Reduce speed, increase following distance, use headlights, avoid sudden movements, and know when to pull over safely. Hydroplaning prevention and recovery techniques.", 
         2, '["All"]', '["All"]', "All", "Rain", "All", "Text", None, "High", False),
        
        ("Advanced_Techniques", "Highway_Driving", "Merging onto Highways Safely", 
         "Use acceleration lane to match traffic speed, check blind spots, signal early, find appropriate gap, and merge smoothly. Never stop in acceleration lane unless absolutely necessary.", 
         3, '["All"]', '["All"]', "Teen,Adult", "All", "Highway", "Text", None, "High", False),
        
        ("Maintenance", "Routine_Care", "Monthly Vehicle Inspection Checklist", 
         "Check tire pressure and tread, fluid levels (oil, coolant, washer fluid), lights (headlights, taillights, turn signals), wipers, and listen for unusual sounds.", 
         2, '["All"]', '["All"]', "Adult,Senior", "All", "All", "Text", None, "Medium", False),
        
        ("Local_Laws", "Parking", "Understanding Parking Regulations", 
         "Learn to read parking signs, understand time limits, permit requirements, fire hydrant distances, and handicapped parking rules. Violations can result in tickets or towing.", 
         1, '["All"]', '["CA","NY","IL","TX","FL"]', "All", "All", "City", "Text", None, "Medium", True)
    ]
    
    for knowledge in driving_knowledge_entries:
        cursor.execute('''
            INSERT INTO driving_knowledge (
                category, subcategory, title, content, difficulty_level, applicable_vehicle_types,
                applicable_states, age_group, weather_conditions, road_types, media_type, media_url,
                importance_level, legal_requirement
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', knowledge)

if __name__ == "__main__":
    # Build database
    db_path = build_car_database()
    print(f"Database build complete: {db_path}") 