"""
Car Order, Customer Service, and Inventory Browser Tool
Comprehensive tool for managing car orders, refunds, repairs, customer service, and inventory browsing
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
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

@register_tool(
    "Comprehensive car order management, customer service, and inventory browsing including order creation, refunds, order support, repair services, customer service, and real inventory browsing",
    [
        {
            "name": "service_type",
            "type": "str", 
            "description": "Type of service: 'create_order', 'refund', 'order_support', 'repair_service', 'customer_service', 'browse_inventory'",
            "prompt": "What type of service do you need?",
            "required": True,
        },
        {
            "name": "car_model",
            "type": "str",
            "description": "Car model (for orders, repairs, and inventory browsing)",
            "prompt": "Which car model?",
            "required": False,
        },
        {
            "name": "dealer_name",
            "type": "str",
            "description": "Dealer name (for orders)",
            "prompt": "Which dealer?",
            "required": False,
        },
        {
            "name": "total_price",
            "type": "str",
            "description": "Total price (for orders)",
            "prompt": "What is the total price?",
            "required": False,
        },
        {
            "name": "customer_id",
            "type": "str",
            "description": "Customer ID",
            "prompt": "Your customer ID?",
            "required": False,
        },
        {
            "name": "order_id",
            "type": "str",
            "description": "Order ID (for refunds and support)",
            "prompt": "Your order ID?",
            "required": False,
        },
        {
            "name": "description",
            "type": "str",
            "description": "Description of issue/reason/inquiry or vehicle criteria for inventory browsing",
            "prompt": "Please describe your issue/request or vehicle criteria?",
            "required": False,
        },
        {
            "name": "location",
            "type": "str",
            "description": "Location (for repair services and inventory browsing)",
            "prompt": "Your location?",
            "required": False,
        },
        {
            "name": "priority",
            "type": "str",
            "description": "Priority level: 'low', 'standard', 'high', 'urgent'",
            "prompt": "Priority level?",
            "required": False,
        },
        {
            "name": "max_price",
            "type": "str",
            "description": "Maximum price budget in USD (for inventory browsing)",
            "prompt": "What's your maximum budget?",
            "required": False,
        },
        {
            "name": "condition_preference",
            "type": "str",
            "description": "Vehicle condition preference: New, Used, Certified_Pre_Owned, or Any (for inventory browsing)",
            "prompt": "Vehicle condition preference?",
            "required": False,
        },
        {
            "name": "color_preference",
            "type": "str",
            "description": "Preferred exterior color (for inventory browsing)",
            "prompt": "Any color preference?",
            "required": False,
        }
    ],
    [
        {
            "name": "service_response",
            "type": "str",
            "description": "Service response in JSON format",
        }
    ],
)
def car_order(service_type: str, **kwargs) -> str:
    """Comprehensive car order, customer service, and inventory browsing tool"""
    try:
        if service_type == "create_order":
            return _create_order(**kwargs)
        elif service_type == "refund":
            return _handle_refund(**kwargs)
        elif service_type == "order_support":
            return _order_support(**kwargs)
        elif service_type == "repair_service":
            return _repair_service(**kwargs)
        elif service_type == "customer_service":
            return _customer_service(**kwargs)
        elif service_type == "browse_inventory":
            return _browse_inventory(**kwargs)
        else:
            error_result = {
                "status": "error",
                "message": f"Unknown service type: {service_type}. Available types: create_order, refund, order_support, repair_service, customer_service, browse_inventory"
            }
            return json.dumps(error_result, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Error in car_order service: {str(e)}")
        error_result = {
            "status": "error",
            "message": f"Service error: {str(e)}"
        }
        return json.dumps(error_result, indent=2, ensure_ascii=False)

def _create_order(car_model: str = "", dealer_name: str = "", total_price: str = "", **kwargs) -> str:
    """Create car purchase order guide"""
    logger.info(f"Creating order guide: {car_model}, dealer: {dealer_name}")
    
    # Parse total price
    try:
        price = float(total_price.replace('$', '').replace(',', ''))
    except:
        price = 30000  # Default price
    
    # Generate order ID
    order_id = f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Order process steps
    order_steps = [
        {
            "step": 1,
            "name": "Confirm Model and Configuration",
            "description": "Confirm final vehicle model, color and configuration with sales",
            "estimated_time": "30 minutes",
            "status": "pending"
        },
        {
            "step": 2,
            "name": "Price Confirmation and Contract Signing",
            "description": "Confirm final price and sign purchase contract",
            "estimated_time": "1 hour",
            "status": "pending"
        },
        {
            "step": 3,
            "name": "Payment Arrangement",
            "description": "Arrange funds according to payment method",
            "estimated_time": "2-5 business days",
            "status": "pending"
        },
        {
            "step": 4,
            "name": "Insurance Purchase",
            "description": "Purchase liability and comprehensive insurance",
            "estimated_time": "1 hour",
            "status": "pending"
        },
        {
            "step": 5,
            "name": "Vehicle Inspection",
            "description": "Check vehicle exterior, interior and functions",
            "estimated_time": "1 hour",
            "status": "pending"
        },
        {
            "step": 6,
            "name": "Registration and Title",
            "description": "Process vehicle registration and title at DMV",
            "estimated_time": "Half day",
            "status": "pending"
        }
    ]
    
    # Cost breakdown
    cost_breakdown = {
        "vehicle_price": price * 0.85,
        "sales_tax": price * 0.08,
        "registration_fees": 350,
        "insurance": 1200,
        "documentation_fees": price * 0.015,
        "total": price
    }
    
    estimated_completion = datetime.now() + timedelta(days=14)
    
    order_guide = {
        "status": "success",
        "service_type": "create_order",
        "order_info": {
            "order_id": order_id,
            "car_model": car_model,
            "dealer_name": dealer_name,
            "total_price": price,
            "created_date": datetime.now().isoformat(),
            "estimated_completion": estimated_completion.strftime("%Y-%m-%d")
        },
        "order_steps": order_steps,
        "cost_breakdown": cost_breakdown,
        "required_documents": {
            "personal": ["Driver's License original and copy", "State ID"],
            "financial": ["Bank Card", "Income Proof (financing)", "Down Payment"],
            "other": ["Proof of Residence (if out-of-state)"]
        },
        "important_notes": [
            "Carefully check contract terms, especially price, configuration and delivery time",
            "Keep all purchase-related receipts and documents",
            "Carefully inspect exterior and functions during vehicle inspection"
        ]
    }
    
    return json.dumps(order_guide, indent=2, ensure_ascii=False)

def handle_refund_request(customer_id: str = "", order_id: str = "", description: str = "", car_model: str = "", **kwargs) -> Dict[str, Any]:
    """Enhanced refund request handler with comprehensive processing logic.
    
    This function provides detailed refund processing including eligibility checking,
    required documentation, timelines, and specific next steps.
    
    Args:
        customer_id (str): Customer identification
        order_id (str): Order reference number
        description (str): Reason for refund request
        car_model (str): Vehicle model
        **kwargs: Additional parameters
        
    Returns:
        Dict[str, Any]: Comprehensive refund processing information
    """
    request_id = f"REF_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Determine refund type and eligibility based on description
    refund_type = "general"
    eligibility_status = "under_review"
    
    # Analyze refund reason
    if any(word in description.lower() for word in ["lemon", "defect", "warranty", "mechanical"]):
        refund_type = "lemon_law"
        eligibility_status = "likely_eligible"
    elif any(word in description.lower() for word in ["misrepresentation", "fraud", "deception", "lied"]):
        refund_type = "misrepresentation"
        eligibility_status = "requires_investigation"
    elif any(word in description.lower() for word in ["changed mind", "no longer", "different car"]):
        refund_type = "buyer_remorse"
        eligibility_status = "limited_eligibility"
    elif any(word in description.lower() for word in ["financing", "loan", "credit", "payment"]):
        refund_type = "financing_issue"
        eligibility_status = "conditional_eligibility"
    
    # Eligibility information
    eligibility_info = {
        "lemon_law": {
            "status": "Strong eligibility under Lemon Law",
            "requirements": "Must demonstrate repeated repair attempts for same issue",
            "timeline": "Typically resolved within 30-45 days",
            "success_rate": "85%"
        },
        "misrepresentation": {
            "status": "Requires evidence of dealer misrepresentation",
            "requirements": "Documentation of false claims or omitted information",
            "timeline": "Investigation period 14-21 days",
            "success_rate": "70%"
        },
        "buyer_remorse": {
            "status": "Limited eligibility - depends on dealer policy",
            "requirements": "Must be within return period (usually 3-7 days)",
            "timeline": "Quick resolution if within policy",
            "success_rate": "30%"
        },
        "financing_issue": {
            "status": "Conditional based on financing terms",
            "requirements": "Must demonstrate financing issue not caused by buyer",
            "timeline": "10-15 business days",
            "success_rate": "60%"
        },
        "general": {
            "status": "Case-by-case evaluation",
            "requirements": "All relevant documentation and evidence",
            "timeline": "Standard processing 7-14 days",
            "success_rate": "45%"
        }
    }
    
    current_eligibility = eligibility_info.get(refund_type, eligibility_info["general"])
    
    # Required documentation specific to refund type
    required_docs = {
        "essential": [
            "Original purchase agreement and all addendums",
            "Payment receipts and financing documents",
            "Government-issued photo ID",
            "Vehicle registration and title (if transferred)"
        ],
        "supporting": []
    }
    
    if refund_type == "lemon_law":
        required_docs["supporting"].extend([
            "All repair orders and invoices",
            "Warranty claim documentation",
            "Written communications with dealer/manufacturer",
            "Independent mechanic assessment (if available)"
        ])
    elif refund_type == "misrepresentation":
        required_docs["supporting"].extend([
            "Original advertisements or marketing materials",
            "Sales representative communications",
            "Vehicle history report discrepancies",
            "Expert evaluation (if applicable)"
        ])
    elif refund_type == "financing_issue":
        required_docs["supporting"].extend([
            "Loan denial letters",
            "Credit report",
            "Alternative financing offers",
            "Income/employment verification"
        ])
    
    # Calculate estimated refund amount
    estimated_refund = {
        "vehicle_value": "Full purchase price minus usage depreciation",
        "fees_recoverable": ["Documentation fees", "Extended warranties", "Unused insurance premiums"],
        "non_recoverable": ["Registration fees", "Used depreciation", "Wear and tear"],
        "estimated_percentage": {
            "lemon_law": "95-100%",
            "misrepresentation": "90-100%", 
            "financing_issue": "85-95%",
            "buyer_remorse": "80-90%",
            "general": "70-90%"
        }.get(refund_type, "70-90%")
    }
    
    # Detailed next steps
    immediate_actions = [
        f"Save your refund request ID: {request_id}",
        "Gather all required documentation listed below",
        "Stop driving the vehicle if safety concerns exist",
        "Contact your insurance company about coverage suspension"
    ]
    
    within_24_hours = [
        "Call our refund hotline: 1-800-REFUND1 (1-800-738-8631)",
        "Email scanned documents to refunds@caradvisor.com",
        "Schedule vehicle inspection appointment if required",
        "Notify your bank/lender about pending refund"
    ]
    
    within_week = [
        "Complete vehicle return inspection",
        "Submit any additional requested documentation",
        "Respond to any follow-up questions from case manager",
        "Keep detailed records of all communications"
    ]
    
    return {
        "status": "success",
        "service_type": "enhanced_refund",
        "request_info": {
            "request_id": request_id,
            "customer_id": customer_id,
            "order_id": order_id,
            "vehicle_model": car_model,
            "refund_reason": description,
            "refund_type": refund_type,
            "eligibility_status": eligibility_status,
            "created_date": datetime.now().isoformat(),
            "estimated_completion": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        },
        "eligibility_assessment": current_eligibility,
        "estimated_refund": estimated_refund,
        "required_documentation": required_docs,
        "action_timeline": {
            "immediate": immediate_actions,
            "within_24_hours": within_24_hours,
            "within_week": within_week
        },
        "contact_information": {
            "primary_phone": "1-800-REFUND1 (1-800-738-8631)",
            "email": "refunds@caradvisor.com",
            "case_manager": "Sarah Johnson, Senior Refund Specialist",
            "direct_extension": "ext. 4567",
            "hours": "Monday-Friday 8AM-6PM, Saturday 9AM-3PM"
        },
        "important_notes": [
            f"Your case type: {refund_type} - {current_eligibility['success_rate']} typical success rate",
            "Refund processing begins immediately upon document completion",
            "Vehicle must remain in substantially same condition as purchased",
            "Legal consultation available for complex cases",
            "All communications will be documented for your protection"
        ],
        "legal_information": {
            "lemon_law_rights": "You may be entitled to a full refund under state Lemon Laws",
            "arbitration_option": "Alternative dispute resolution available",
            "legal_counsel": "You have the right to legal representation",
            "time_limits": "Important deadlines may apply - act promptly"
        }
    }

def _handle_refund(customer_id: str = "", order_id: str = "", description: str = "", car_model: str = "", **kwargs) -> str:
    """Handle refund requests with enhanced processing"""
    logger.info(f"Processing refund request: Order {order_id}, Customer {customer_id}")
    
    # Use enhanced refund processing
    refund_data = handle_refund_request(customer_id, order_id, description, car_model, **kwargs)
    
    return json.dumps(refund_data, indent=2, ensure_ascii=False)

def _order_support(order_id: str = "", customer_id: str = "", description: str = "", **kwargs) -> str:
    """Handle order support issues"""
    case_id = f"CASE_{customer_id}_{order_id}" if order_id else f"CASE_{customer_id}_GENERAL"
    
    # Determine issue type from description
    issue_type = "general"
    if "missing" in description.lower():
        issue_type = "missing_order"
    elif "delay" in description.lower():
        issue_type = "delivery_delay"
    elif "incorrect" in description.lower():
        issue_type = "incorrect_order"
    
    if issue_type == "missing_order":
        support_response = {
            "status": "success",
            "service_type": "order_support",
            "case_info": {
                "case_id": case_id,
                "issue_type": issue_type,
                "status": "Investigating",
                "priority": "High",
                "estimated_resolution": "24-48 hours"
            },
            "investigation_steps": [
                "Checking order database for records",
                "Verifying payment processing",
                "Contacting fulfillment centers",
                "Dealer network inquiry"
            ],
            "compensation": "Priority processing for replacement order",
            "contact_info": {
                "phone": "1-800-ORDERS1",
                "email": "orders@caradvisor.com",
                "specialist": "Michael Chen"
            }
        }
    else:
        support_response = {
            "status": "success",
            "service_type": "order_support",
            "case_info": {
                "case_id": case_id,
                "issue_type": issue_type,
                "status": "Open",
                "priority": "High" if "urgent" in description.lower() else "Standard",
                "assigned_agent": "Michael Chen",
                "estimated_resolution": "2-3 business days"
            },
            "issue_details": {
                "description": description,
                "order_id": order_id
            },
            "next_steps": [
                "Specialist review within 2 hours",
                "Detailed action plan via email",
                "Regular updates until resolution"
            ],
            "contact_info": {
                "phone": "1-800-ORDERS1",
                "email": "orders@caradvisor.com",
                "assigned_agent": "Michael Chen"
            }
        }
    
    return json.dumps(support_response, indent=2, ensure_ascii=False)

def _repair_service(car_model: str = "", description: str = "", location: str = "", **kwargs) -> str:
    """Handle repair service requests"""
    warranty_status = kwargs.get("warranty_status", "unknown")
    
    # Sample service centers
    service_centers = [
        {
            "name": "AutoCare Service Center",
            "address": f"123 Main St, {location}",
            "phone": "(555) 123-4567",
            "specialties": ["Engine repair", "Transmission", "Electrical"],
            "rating": 4.8,
            "estimated_cost": "$150-400"
        },
        {
            "name": "Premium Vehicle Services", 
            "address": f"456 Oak Ave, {location}",
            "phone": "(555) 987-6543",
            "specialties": ["Luxury vehicles", "Advanced diagnostics", "Warranty work"],
            "rating": 4.9,
            "estimated_cost": "$200-500"
        }
    ]
    
    warranty_info = {
        "active": {
            "coverage": "Full coverage under manufacturer warranty",
            "estimated_cost": "$0-50 (deductible only)",
            "recommendation": "Visit authorized dealer service center"
        },
        "expired": {
            "coverage": "No warranty coverage",
            "estimated_cost": "$150-800 depending on issue",
            "recommendation": "Compare prices between authorized and independent shops"
        },
        "unknown": {
            "coverage": "Warranty status needs verification",
            "estimated_cost": "$0-800 depending on warranty status",
            "recommendation": "Check warranty status first, then proceed"
        }
    }
    
    current_warranty = warranty_info.get(warranty_status, warranty_info["unknown"])
    
    repair_response = {
        "status": "success",
        "service_type": "repair_service",
        "vehicle_info": {
            "model": car_model,
            "issue": description,
            "location": location
        },
        "warranty_status": {
            "status": warranty_status,
            "coverage": current_warranty["coverage"],
            "estimated_cost": current_warranty["estimated_cost"],
            "recommendation": current_warranty["recommendation"]
        },
        "service_centers": service_centers,
        "next_steps": [
            "Verify warranty status if unknown",
            "Get diagnostic assessment",
            "Compare quotes from multiple centers",
            "Schedule appointment"
        ]
    }
    
    return json.dumps(repair_response, indent=2, ensure_ascii=False)

def _customer_service(description: str = "", customer_id: str = "", priority: str = "standard", **kwargs) -> str:
    """Handle general customer service inquiries"""
    
    # Determine inquiry type from description
    inquiry_type = "question"
    if any(word in description.lower() for word in ["complaint", "complain", "dissatisfied", "unhappy"]):
        inquiry_type = "complaint"
    elif any(word in description.lower() for word in ["feedback", "suggestion", "improve"]):
        inquiry_type = "feedback"
    elif any(word in description.lower() for word in ["technical", "app", "website", "login", "system"]):
        inquiry_type = "technical_issue"
    
    ticket_id = f"CS_{customer_id}_{inquiry_type}_{hash(description) % 10000:04d}"
    
    response_templates = {
        "complaint": {
            "message": "We sincerely apologize for your experience. Your complaint is very important to us.",
            "escalation": "Manager review within 4 hours",
            "resolution_time": "24-48 hours"
        },
        "question": {
            "message": "Thank you for your inquiry. We're happy to help answer your questions.",
            "escalation": "Specialist consultation if needed",
            "resolution_time": "Same day"
        },
        "feedback": {
            "message": "We appreciate your feedback. Your input helps us improve our services.",
            "escalation": "Product team review",
            "resolution_time": "5-7 business days"
        },
        "technical_issue": {
            "message": "We understand technical issues can be frustrating. Our tech team will assist you.",
            "escalation": "Technical specialist assignment",
            "resolution_time": "2-4 hours"
        }
    }
    
    template = response_templates.get(inquiry_type, response_templates["question"])
    
    # Priority adjustments
    if priority == "urgent":
        resolution_time = "1-2 hours"
    elif priority == "high":
        resolution_time = "4-6 hours"
    else:
        resolution_time = template["resolution_time"]
    
    customer_service_response = {
        "status": "success",
        "service_type": "customer_service",
        "ticket_info": {
            "ticket_id": ticket_id,
            "inquiry_type": inquiry_type,
            "priority": priority,
            "status": "Open",
            "estimated_resolution": resolution_time,
            "assigned_department": "Customer Success Team"
        },
        "inquiry_details": {
            "description": description,
            "customer_id": customer_id
        },
        "response_info": {
            "message": template["message"],
            "escalation": template["escalation"]
        },
        "contact_info": {
            "phone": "1-800-SUPPORT",
            "email": "support@caradvisor.com",
            "hours": "24/7 for urgent issues, 8AM-8PM for standard inquiries"
        },
        "next_steps": [
            template["escalation"],
            f"Estimated Resolution: {resolution_time}",
            "Email updates on progress",
            "Satisfaction survey after resolution"
        ]
    }
    
    return json.dumps(customer_service_response, indent=2, ensure_ascii=False)

def _browse_inventory(description: str = "", car_model: str = "", max_price: str = "", location: str = "", 
                     condition_preference: str = "Any", color_preference: str = "", **kwargs) -> str:
    """Browse real vehicle inventory with VINs and immediate availability"""
    
    try:
        logger.info(f"Browsing inventory for: {description or car_model}")
        
        # Use description as vehicle_criteria if provided, otherwise use car_model
        vehicle_criteria = description or car_model or "any vehicle"
        
        # Parse price filter
        price_limit = None
        if max_price:
            try:
                price_limit = float(max_price.replace('$', '').replace(',', ''))
            except ValueError:
                pass
        
        # Parse vehicle criteria
        criteria_words = vehicle_criteria.lower().split()
        brand_filter = None
        model_filter = None
        year_filter = None
        body_type_filter = None
        
        # Extract search terms
        for word in criteria_words:
            if word.isdigit() and 2020 <= int(word) <= 2030:
                year_filter = int(word)
            elif word in ['suv', 'sedan', 'pickup', 'truck', 'coupe', 'convertible', 'hatchback', 'wagon']:
                body_type_filter = word.title()
        
        # Common brand detection
        brands = ['toyota', 'honda', 'ford', 'chevrolet', 'bmw', 'mercedes', 'audi', 'lexus', 'acura', 'nissan', 'mazda', 'subaru', 'tesla', 'volvo', 'infiniti', 'cadillac', 'lincoln', 'buick', 'gmc', 'dodge', 'jeep', 'ram', 'chrysler', 'hyundai', 'kia', 'genesis', 'porsche', 'jaguar', 'land rover', 'mini', 'volkswagen', 'mitsubishi']
        for word in criteria_words:
            if word in brands:
                brand_filter = word.title()
                break
        
        # Model detection
        models = ['camry', 'civic', 'accord', 'corolla', 'f-150', 'silverado', 'rav4', 'cr-v', 'pilot', 'highlander', 'prius', 'mustang', 'explorer', 'escape', 'tahoe', 'suburban', 'malibu', 'equinox', 'altima', 'sentra', 'rogue', 'pathfinder', 'cx-5', 'mazda3', 'mazda6', 'outback', 'forester', 'legacy', 'model-3', 'model-s', 'model-x', 'model-y']
        for word in criteria_words:
            if word in models:
                model_filter = word.replace('-', ' ').title()
                break
        
        # Build database query
        conn = _get_database_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT 
            i.id, i.vin, i.stock_number, i.condition_type, i.exterior_color, i.interior_color,
            i.trim_level, i.mileage, i.model_year, i.asking_price, i.invoice_price, i.market_adjustment,
            i.incentives_available, i.status, i.days_on_lot, i.acquisition_date, i.expected_arrival_date,
            i.carfax_report_available, i.accident_history, i.previous_owners, i.service_records_available,
            i.key_count, i.special_notes, i.photos_url, i.video_url, i.featured_listing, i.online_price,
            i.certified_pre_owned, i.warranty_remaining_months, i.financing_specials, i.lease_specials,
            d.id, d.name, d.phone, d.city, d.state,
            v.id, v.brand, v.model, v.year, v.body_type, v.fuel_type, v.transmission, v.drivetrain,
            v.msrp, v.engine_type, v.horsepower, v.torque, v.acceleration_0_60, v.mpg_city, v.mpg_highway,
            v.seating_capacity, v.cargo_space, v.safety_rating
        FROM inventory i
        JOIN dealers d ON i.dealer_id = d.id
        JOIN vehicles v ON i.vehicle_id = v.id
        WHERE i.status IN ('Available', 'In_Transit')
        """
        
        params = []
        
        # Add filters
        if brand_filter:
            query += " AND LOWER(v.brand) = LOWER(?)"
            params.append(brand_filter)
        
        if model_filter:
            query += " AND LOWER(v.model) LIKE LOWER(?)"
            params.append(f"%{model_filter}%")
        
        if year_filter:
            query += " AND v.year = ?"
            params.append(year_filter)
        
        if body_type_filter:
            query += " AND LOWER(v.body_type) = LOWER(?)"
            params.append(body_type_filter)
        
        if price_limit:
            query += " AND i.asking_price <= ?"
            params.append(price_limit)
        
        if condition_preference and condition_preference.lower() != "any":
            condition_map = {
                "new": "New",
                "used": "Used", 
                "certified pre-owned": "Certified_Pre_Owned",
                "certified": "Certified_Pre_Owned",
                "cpo": "Certified_Pre_Owned"
            }
            condition_db = condition_map.get(condition_preference.lower(), condition_preference)
            query += " AND i.condition_type = ?"
            params.append(condition_db)
        
        if color_preference:
            query += " AND LOWER(i.exterior_color) LIKE LOWER(?)"
            params.append(f"%{color_preference}%")
        
        if location:
            query += " AND (LOWER(d.city) LIKE LOWER(?) OR LOWER(d.state) LIKE LOWER(?))"
            params.extend([f"%{location}%", f"%{location}%"])
        
        # Order by featured listings first, then by days on lot (negotiation opportunity)
        query += " ORDER BY i.featured_listing DESC, i.days_on_lot DESC, i.asking_price ASC LIMIT 20"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            conn.close()
            return json.dumps({
                "status": "no_results",
                "service_type": "browse_inventory",
                "message": "No vehicles found matching your criteria",
                "suggestions": [
                    "Try broader search terms (e.g., just brand name)",
                    "Increase your budget range",
                    "Consider different locations",
                    "Try 'Any' for condition preference",
                    "Remove color restrictions"
                ],
                "search_criteria": {
                    "original_query": vehicle_criteria,
                    "filters_applied": {
                        "brand": brand_filter,
                        "model": model_filter,
                        "year": year_filter,
                        "body_type": body_type_filter,
                        "max_price": price_limit,
                        "condition": condition_preference,
                        "color": color_preference,
                        "location": location
                    }
                }
            }, indent=2, ensure_ascii=False)
        
        # Format results
        inventory_vehicles = []
        for vehicle_data in results:
            formatted_vehicle = _format_vehicle_details(vehicle_data)
            inventory_vehicles.append(formatted_vehicle)
        
        conn.close()
        
        # Generate summary and recommendations
        summary = _generate_inventory_summary(inventory_vehicles)
        recommendations = _generate_purchase_recommendations(inventory_vehicles)
        
        response = {
            "status": "success",
            "service_type": "browse_inventory",
            "search_summary": {
                "total_vehicles_found": len(inventory_vehicles),
                "search_criteria": vehicle_criteria,
                "filters_applied": {
                    "brand": brand_filter,
                    "model": model_filter,
                    "year": year_filter,
                    "body_type": body_type_filter,
                    "max_price": price_limit,
                    "condition": condition_preference,
                    "color": color_preference,
                    "location": location
                },
                "price_range": {
                    "lowest": min(v["pricing"]["asking_price"] for v in inventory_vehicles if v["pricing"]["asking_price"]),
                    "highest": max(v["pricing"]["asking_price"] for v in inventory_vehicles if v["pricing"]["asking_price"]),
                    "average": sum(v["pricing"]["asking_price"] for v in inventory_vehicles if v["pricing"]["asking_price"]) / len(inventory_vehicles)
                }
            },
            "inventory": inventory_vehicles,
            "market_insights": summary,
            "recommendations": recommendations,
            "next_steps": [
                "Contact dealers to schedule test drives",
                "Verify current availability (inventory moves fast)",
                "Request additional photos or videos",
                "Ask about financing pre-approval",
                "Negotiate based on days on lot",
                "Schedule professional inspection for used vehicles"
            ]
        }
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error browsing inventory: {str(e)}")
        return json.dumps({
            "status": "error",
            "service_type": "browse_inventory",
            "message": f"Failed to browse inventory: {str(e)}",
            "suggestions": [
                "Check your search criteria format",
                "Ensure the car database is available",
                "Try a simpler search query"
            ]
        }, indent=2, ensure_ascii=False)

def _format_vehicle_details(vehicle_data: tuple) -> Dict[str, Any]:
    """Format raw vehicle data into structured response"""
    (
        inv_id, vin, stock_number, condition_type, exterior_color, interior_color,
        trim_level, mileage, model_year, asking_price, invoice_price, market_adjustment,
        incentives_available, status, days_on_lot, acquisition_date, expected_arrival_date,
        carfax_report_available, accident_history, previous_owners, service_records_available,
        key_count, special_notes, photos_url, video_url, featured_listing, online_price,
        certified_pre_owned, warranty_remaining_months, financing_specials, lease_specials,
        dealer_id, dealer_name, dealer_phone, dealer_city, dealer_state,
        vehicle_id, brand, model, year, body_type, fuel_type, transmission, drivetrain,
        msrp, engine_type, horsepower, torque, acceleration_0_60, mpg_city, mpg_highway,
        seating_capacity, cargo_space, safety_rating
    ) = vehicle_data
    
    # Calculate value indicators
    msrp_difference = asking_price - msrp if msrp and asking_price else 0
    days_available = days_on_lot or 0
    
    # Determine urgency and negotiation opportunity
    urgency_level = "High" if days_available > 60 else "Medium" if days_available > 30 else "Low"
    negotiation_potential = "High" if days_available > 45 or msrp_difference > 2000 else "Medium" if days_available > 20 else "Low"
    
    # Parse JSON fields safely
    try:
        photos = json.loads(photos_url) if photos_url else []
    except:
        photos = [photos_url] if photos_url else []
    
    try:
        incentives = json.loads(incentives_available) if incentives_available else []
    except:
        incentives = [incentives_available] if incentives_available else []
    
    return {
        "inventory_id": inv_id,
        "vehicle_identification": {
            "vin": vin,
            "stock_number": stock_number,
            "condition": condition_type,
            "model_year": model_year
        },
        "vehicle_details": {
            "brand": brand,
            "model": model,
            "year": year,
            "trim_level": trim_level,
            "body_type": body_type,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "drivetrain": drivetrain
        },
        "appearance": {
            "exterior_color": exterior_color,
            "interior_color": interior_color
        },
        "condition_info": {
            "mileage": mileage,
            "accident_history": bool(accident_history),
            "previous_owners": previous_owners or 0,
            "service_records_available": bool(service_records_available),
            "carfax_report_available": bool(carfax_report_available)
        },
        "specifications": {
            "engine_type": engine_type,
            "horsepower": horsepower,
            "torque": torque,
            "acceleration_0_60": acceleration_0_60,
            "mpg_city": mpg_city,
            "mpg_highway": mpg_highway,
            "seating_capacity": seating_capacity,
            "cargo_space": cargo_space,
            "safety_rating": safety_rating
        },
        "pricing": {
            "asking_price": asking_price,
            "msrp": msrp,
            "invoice_price": invoice_price,
            "market_adjustment": market_adjustment,
            "online_price": online_price,
            "price_vs_msrp": msrp_difference,
            "financing_specials": financing_specials,
            "lease_specials": lease_specials,
            "current_incentives": incentives
        },
        "availability": {
            "status": status,
            "days_on_lot": days_available,
            "acquisition_date": acquisition_date,
            "expected_arrival_date": expected_arrival_date,
            "urgency_level": urgency_level,
            "negotiation_potential": negotiation_potential
        },
        "dealer_info": {
            "dealer_id": dealer_id,
            "name": dealer_name,
            "phone": dealer_phone,
            "location": f"{dealer_city}, {dealer_state}"
        },
        "additional_info": {
            "certified_pre_owned": bool(certified_pre_owned),
            "warranty_remaining_months": warranty_remaining_months,
            "key_count": key_count,
            "special_notes": special_notes,
            "featured_listing": bool(featured_listing),
            "photos": photos,
            "video_url": video_url
        },
        "purchase_readiness": {
            "immediate_availability": status == "Available",
            "test_drive_ready": key_count >= 1 and status == "Available",
            "documentation_complete": bool(carfax_report_available),
            "financing_options": bool(financing_specials),
            "overall_score": _calculate_purchase_readiness_score(
                status, key_count, carfax_report_available, financing_specials, days_available
            )
        }
    }

def _calculate_purchase_readiness_score(status: str, key_count: int, carfax: bool, financing: str, days_on_lot: int) -> int:
    """Calculate a purchase readiness score (1-10)"""
    score = 5  # Base score
    
    if status == "Available":
        score += 2
    elif status == "Reserved":
        score -= 3
    
    if key_count >= 2:
        score += 1
    elif key_count >= 1:
        score += 0.5
    
    if carfax:
        score += 1
    
    if financing:
        score += 1
    
    # More days on lot = better negotiation opportunity
    if days_on_lot > 60:
        score += 1
    elif days_on_lot > 30:
        score += 0.5
    
    return min(10, max(1, int(score)))

def _generate_inventory_summary(vehicles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate market insights from available inventory"""
    if not vehicles:
        return {}
    
    # Condition breakdown
    conditions = {}
    for v in vehicles:
        condition = v["vehicle_identification"]["condition"]
        conditions[condition] = conditions.get(condition, 0) + 1
    
    # Color popularity
    colors = {}
    for v in vehicles:
        color = v["appearance"]["exterior_color"]
        colors[color] = colors.get(color, 0) + 1
    
    # Negotiation opportunities
    high_negotiation = [v for v in vehicles if v["availability"]["negotiation_potential"] == "High"]
    
    # Price vs MSRP analysis
    overpriced = [v for v in vehicles if v["pricing"]["price_vs_msrp"] > 1000]
    underpriced = [v for v in vehicles if v["pricing"]["price_vs_msrp"] < -1000]
    
    return {
        "condition_breakdown": conditions,
        "color_availability": colors,
        "negotiation_opportunities": len(high_negotiation),
        "pricing_analysis": {
            "vehicles_over_msrp": len(overpriced),
            "vehicles_under_msrp": len(underpriced),
            "average_markup": sum(v["pricing"]["price_vs_msrp"] for v in vehicles) / len(vehicles)
        },
        "inventory_freshness": {
            "newly_arrived": len([v for v in vehicles if v["availability"]["days_on_lot"] <= 7]),
            "aged_inventory": len([v for v in vehicles if v["availability"]["days_on_lot"] > 60])
        }
    }

def _generate_purchase_recommendations(vehicles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate smart purchase recommendations"""
    if not vehicles:
        return {}
    
    recommendations = {}
    
    # Best value (considering price vs MSRP and features)
    best_value = min(vehicles, key=lambda x: x["pricing"]["price_vs_msrp"])
    recommendations["best_value"] = {
        "vehicle": f"{best_value['vehicle_details']['year']} {best_value['vehicle_details']['brand']} {best_value['vehicle_details']['model']}",
        "reason": f"${abs(best_value['pricing']['price_vs_msrp']):,.0f} {'under' if best_value['pricing']['price_vs_msrp'] < 0 else 'over'} MSRP",
        "vin": best_value["vehicle_identification"]["vin"]
    }
    
    # Best negotiation opportunity
    best_negotiation = max(vehicles, key=lambda x: x["availability"]["days_on_lot"])
    recommendations["best_negotiation"] = {
        "vehicle": f"{best_negotiation['vehicle_details']['year']} {best_negotiation['vehicle_details']['brand']} {best_negotiation['vehicle_details']['model']}",
        "reason": f"{best_negotiation['availability']['days_on_lot']} days on lot - strong negotiation position",
        "vin": best_negotiation["vehicle_identification"]["vin"]
    }
    
    # Highest purchase readiness score
    most_ready = max(vehicles, key=lambda x: x["purchase_readiness"]["overall_score"])
    recommendations["most_ready_to_buy"] = {
        "vehicle": f"{most_ready['vehicle_details']['year']} {most_ready['vehicle_details']['brand']} {most_ready['vehicle_details']['model']}",
        "reason": f"Purchase readiness score: {most_ready['purchase_readiness']['overall_score']}/10",
        "vin": most_ready["vehicle_identification"]["vin"]
    }
    
    # Featured vehicles (dealer priorities)
    featured = [v for v in vehicles if v["additional_info"]["featured_listing"]]
    if featured:
        recommendations["dealer_featured"] = {
            "count": len(featured),
            "reason": "Dealers are actively promoting these vehicles"
        }
    
    return recommendations
