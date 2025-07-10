"""
Car Advisor Worker - User preference analysis and personalized recommendation system

This worker focuses on:
1. Analyzing user needs and preferences
2. Coordinating with CarDatabaseWorker for data queries
3. Integrating database results with personalized insights
4. Providing comprehensive automotive consultation
5. Problem classification and resolution prioritization
"""

import json
import logging
import math
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, START, END
from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.utils.slot import Slot

logger = logging.getLogger(__name__)


@register_worker
class CarAdvisorWorker(BaseWorker):
    """
    Car advisor focused on preference analysis and personalized recommendations
    Works in coordination with CarDatabaseWorker for complete automotive consultation
    Enhanced with problem classification and resolution prioritization
    """
    
    description: str = "Analyze user preferences, coordinate database queries, provide personalized automotive recommendations, and prioritize problem resolution"
    
    def __init__(self) -> None:
        super().__init__()
        self.action_graph: StateGraph = self._create_action_graph()
        self.problem_types = {
            "refund": ["refund", "return", "cancel order", "money back", "cancelled"],
            "repair": ["repair", "fix", "broken", "problem", "issue", "maintenance"],
            "specific_query": ["specific", "particular", "2026", "availability", "when", "model year"],
            "recommendation": ["recommend", "suggest", "looking for", "help me find", "best"],
            "general": []
        }
    
    def classify_user_problem(self, message: str) -> Dict[str, Any]:
        """Classify the user's problem type to prioritize appropriate response.
        
        This method identifies whether the user has a specific problem that needs
        immediate resolution (refund, repair) or is seeking recommendations.
        
        Args:
            message (str): User's input message
            
        Returns:
            Dict[str, Any]: Classification result with type, confidence, and context
        """
        message_lower = message.lower()
        
        # Check for problem patterns with confidence scoring
        for problem_type, keywords in self.problem_types.items():
            if problem_type == "general":
                continue
                
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > 0:
                confidence = min(0.9, 0.6 + (matches * 0.15))  # Cap at 0.9
                
                context = {
                    "requires_immediate_action": problem_type in ["refund", "repair"],
                    "needs_preference_collection": problem_type in ["recommendation", "general"],
                    "specific_info_needed": problem_type == "specific_query",
                    "matched_keywords": [kw for kw in keywords if kw in message_lower]
                }
                
                return {
                    "type": problem_type,
                    "confidence": confidence,
                    "context": context,
                    "priority": "high" if problem_type in ["refund", "repair"] else "medium"
                }
        
        # Default to general with low confidence
        return {
            "type": "general",
            "confidence": 0.3,
            "context": {
                "requires_immediate_action": False,
                "needs_preference_collection": True,
                "specific_info_needed": False,
                "matched_keywords": []
            },
            "priority": "low"
        }
    
    def generate_problem_focused_response(self, problem_classification: Dict[str, Any], message: str) -> str:
        """Generate a response that prioritizes problem resolution with actionable next steps.
        
        Args:
            problem_classification (Dict[str, Any]): Result from classify_user_problem
            message (str): Original user message
            
        Returns:
            str: Problem-focused response with clear next steps
        """
        problem_type = problem_classification["type"]
        context = problem_classification["context"]
        
        if problem_type == "refund":
            return """I understand you need help with a refund. Here are your next steps:

1. **Immediate Action**: I can help verify your order details and refund eligibility
2. **Information Needed**: Your order number, purchase date, and reason for refund
3. **Next Steps**: I'll guide you through the specific refund process and provide direct contact information

Please share your order details, and I'll get this resolved for you right away."""
            
        elif problem_type == "repair":
            return """I'll help you with your vehicle repair needs. Here's how I can assist:

1. **Issue Assessment**: Tell me the specific problem you're experiencing
2. **Vehicle Details**: Share your car's make, model, and when the issue started  
3. **Next Steps**: I'll connect you with certified repair shops in your area and provide cost estimates

What specific issue are you experiencing with your vehicle? I'll get you connected with the right repair services immediately."""
            
        elif problem_type == "specific_query":
            return """I'll provide you with detailed information about that vehicle. Here's what I can help with:

1. **Specifications**: Complete vehicle details and features
2. **Availability**: Current inventory and delivery timelines
3. **Pricing**: Market pricing and available offers
4. **Next Steps**: Connect you with dealers for test drives and purchases

What specific details about the vehicle are you most interested in learning?"""
            
        elif problem_type == "recommendation":
            return """I'll provide personalized vehicle recommendations based on your needs. Let me gather some key information:

1. **Budget**: What's your target price range?
2. **Vehicle Type**: Sedan, SUV, truck, or other preference?
3. **Primary Use**: Daily commuting, family trips, work, or recreation?
4. **Key Features**: Any must-have features or priorities?

Once I understand your preferences, I'll provide tailored recommendations with specific next steps for purchasing."""
            
        else:  # general
            return """I'm here to help with all your automotive needs. I can assist with:

1. **Vehicle Recommendations** based on your preferences and budget
2. **Order Tracking and Support** for existing purchases
3. **Repair Services** and maintenance guidance
4. **Pricing Information** and market analysis

What specific automotive assistance can I provide for you today? I'll give you actionable next steps right away."""
    
    def analyze_user_preferences(self, state: MessageState) -> MessageState:
        """Analyze user needs and preferences from their input"""
        try:
            # Initialize slots if not exists
            if state.slots is None:
                state.slots = {}
            if "car_advisor" not in state.slots:
                state.slots["car_advisor"] = []
            
            # Extract user inputs
            user_inputs = self._extract_user_inputs(state)
            
            # Build comprehensive user profile
            user_profile = self._build_user_preference_profile(user_inputs)
            
            # Store analysis results in slots
            state.slots["car_advisor"] = [
                Slot(
                    name="user_inputs",
                    type="str", 
                    value=json.dumps(user_inputs, ensure_ascii=False),
                    description="Raw user input data"
                ),
                Slot(
                    name="user_profile",
                    type="str",
                    value=json.dumps(user_profile, ensure_ascii=False), 
                    description="Analyzed user preference profile"
                ),
                Slot(
                    name="analysis_stage",
                    type="str",
                    value="preferences_analyzed",
                    description="Current analysis stage"
                )
            ]
            
            # Prepare database query parameters
            db_query_params = self._prepare_database_query_params(user_profile)
            state.slots["car_advisor"].append(
                Slot(
                    name="db_query_params",
                    type="str",
                    value=json.dumps(db_query_params, ensure_ascii=False),
                    description="Parameters for database queries"
                )
            )
            
            logger.info(f"User preferences analyzed: {user_profile.get('user_type', 'Unknown')}")
            state.status = StatusEnum.COMPLETE
            return state
            
        except Exception as e:
            logger.error(f"Error in analyze_user_preferences: {str(e)}")
            state.response = f"Error analyzing preferences: {str(e)}"
            state.status = StatusEnum.INCOMPLETE
            return state
    
    def coordinate_database_queries(self, state: MessageState) -> MessageState:
        """Coordinate with CarDatabaseWorker to get vehicle data"""
        try:
            # Get database query parameters from slots
            db_query_params = {}
            for slot in state.slots.get("car_advisor", []):
                if slot.name == "db_query_params" and slot.value:
                    db_query_params = json.loads(slot.value)
                    break
            
            # Coordinate with CarDatabaseWorker to execute database queries
            database_results = self._coordinate_with_database_worker(state, db_query_params)
            
            # Store database results in slots
            state.slots["car_advisor"].append(
                Slot(
                    name="database_results",
                    type="str",
                    value=json.dumps(database_results, ensure_ascii=False),
                    description="Results from database queries"
                )
            )
            
            # Update analysis stage
            for slot in state.slots.get("car_advisor", []):
                if slot.name == "analysis_stage":
                    slot.value = "database_queried"
                    break
            
            logger.info(f"Database coordination completed: {len(database_results.get('vehicles', []))} vehicles found")
            state.status = StatusEnum.COMPLETE
            return state
            
        except Exception as e:
            logger.error(f"Error in coordinate_database_queries: {str(e)}")
            state.response = f"Error coordinating database queries: {str(e)}"
            state.status = StatusEnum.INCOMPLETE
            return state
    
    def integrate_and_recommend(self, state: MessageState) -> MessageState:
        """Integrate database results with user preferences to create personalized recommendations"""
        try:
            # Get data from slots
            user_profile = {}
            database_results = {}
            user_inputs = {}
            
            for slot in state.slots.get("car_advisor", []):
                if slot.value:
                    if slot.name == "user_profile":
                        user_profile = json.loads(slot.value)
                    elif slot.name == "database_results":
                        database_results = json.loads(slot.value)
                    elif slot.name == "user_inputs":
                        user_inputs = json.loads(slot.value)
            
            # Create personalized recommendations
            recommendations = self._create_personalized_recommendations(
                user_profile, database_results, user_inputs
            )
            
            # Generate comprehensive response
            comprehensive_response = self._generate_comprehensive_response(
                user_profile, database_results, recommendations, user_inputs
            )
            
            # Store final results in slots
            state.slots["car_advisor"].extend([
                Slot(
                    name="recommendations",
                    type="str",
                    value=json.dumps(recommendations, ensure_ascii=False),
                    description="Personalized vehicle recommendations"
                ),
                Slot(
                    name="final_response",
                    type="str", 
                    value=json.dumps(comprehensive_response, ensure_ascii=False),
                    description="Complete consultation response"
                )
            ])
            
            # Set final response
            state.response = self._format_user_response(comprehensive_response)
            state.status = StatusEnum.COMPLETE
            
            logger.info("Personalized recommendations completed")
            return state
            
        except Exception as e:
            logger.error(f"Error in integrate_and_recommend: {str(e)}")
            state.response = f"Error creating recommendations: {str(e)}"
            state.status = StatusEnum.INCOMPLETE
            return state
    
    def _create_action_graph(self) -> StateGraph:
        """Create the coordination workflow graph"""
        workflow = StateGraph(MessageState)
        
        # Add nodes for the advisor workflow
        workflow.add_node("analyze_preferences", self.analyze_user_preferences)
        workflow.add_node("coordinate_database", self.coordinate_database_queries)
        workflow.add_node("integrate_recommend", self.integrate_and_recommend)
        
        # Create linear workflow
        workflow.add_edge(START, "analyze_preferences")
        workflow.add_edge("analyze_preferences", "coordinate_database")
        workflow.add_edge("coordinate_database", "integrate_recommend")
        workflow.add_edge("integrate_recommend", END)
        
        return workflow
    
    def _extract_user_inputs(self, state: MessageState) -> Dict[str, Any]:
        """Extract and categorize user inputs"""
        try:
            # Get user message content
            user_message = ""
            if state.user_message:
                user_message = state.user_message.message if hasattr(state.user_message, 'message') else str(state.user_message)
            
            # Get orchestrator attributes if available
            task_context = ""
            if hasattr(state, 'orchestrator_message') and state.orchestrator_message:
                task_context = getattr(state.orchestrator_message, 'attribute', {}).get('task', '')
            
            combined_input = f"{user_message} {task_context}".strip()
            
            # Categorize the input
            input_category = self._categorize_user_intent(combined_input)
            
            return {
                "raw_message": user_message,
                "task_context": task_context,
                "combined_input": combined_input,
                "input_category": input_category,
                "timestamp": datetime.now().isoformat(),
                "extracted_preferences": self._extract_specific_preferences(combined_input)
            }
            
        except Exception as e:
            logger.error(f"Error extracting user inputs: {str(e)}")
            return {
                "raw_message": "Error parsing input",
                "task_context": "",
                "combined_input": "Error parsing input",
                "input_category": "general_inquiry",
                "timestamp": datetime.now().isoformat(),
                "extracted_preferences": {}
            }
    
    def _categorize_user_intent(self, message: str) -> str:
        """Categorize user intent for appropriate handling"""
        message_lower = message.lower()
        
        if any(term in message_lower for term in ["recommend", "suggest", "looking for", "need", "want"]):
            return "vehicle_recommendation"
        elif any(term in message_lower for term in ["compare", "vs", "versus", "difference"]):
            return "vehicle_comparison"
        elif any(term in message_lower for term in ["dealer", "dealership", "where to buy", "contact"]):
            return "dealer_connection"
        elif any(term in message_lower for term in ["price", "cost", "budget", "affordable", "market"]):
            return "pricing_inquiry"
        elif any(term in message_lower for term in ["order", "track", "status", "delivery"]):
            return "order_tracking"
        elif any(term in message_lower for term in ["repair", "service", "maintenance", "issue", "problem"]):
            return "service_inquiry"
        else:
            return "general_consultation"
    
    def _extract_specific_preferences(self, message: str) -> Dict[str, Any]:
        """Extract specific automotive preferences from message"""
        message_lower = message.lower()
        preferences = {}
        
        # Vehicle type preferences
        vehicle_types = {
            'sedan': ['sedan', 'car'],
            'suv': ['suv', 'crossover'],
            'truck': ['truck', 'pickup'],
            'coupe': ['coupe', 'sports car'],
            'convertible': ['convertible', 'cabriolet'],
            'wagon': ['wagon', 'estate'],
            'hatchback': ['hatchback', 'hatch']
        }
        
        for v_type, keywords in vehicle_types.items():
            if any(keyword in message_lower for keyword in keywords):
                preferences['vehicle_type'] = v_type
                break
        
        # Brand preferences
        brands = ['toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi', 'chevrolet', 
                 'nissan', 'hyundai', 'kia', 'mazda', 'subaru', 'volkswagen', 'tesla',
                 'lexus', 'acura', 'infiniti', 'cadillac', 'lincoln', 'buick']
        
        for brand in brands:
            if brand in message_lower:
                preferences['preferred_brand'] = brand.title()
                break
        
        # Fuel type preferences
        if any(term in message_lower for term in ['electric', 'ev', 'battery']):
            preferences['fuel_type'] = 'Electric'
        elif any(term in message_lower for term in ['hybrid', 'gas-electric']):
            preferences['fuel_type'] = 'Hybrid'
        elif any(term in message_lower for term in ['gas', 'gasoline', 'petrol']):
            preferences['fuel_type'] = 'Gasoline'
        
        # Budget extraction
        price_patterns = re.findall(r'\$(\d{1,3}(?:,\d{3})*)', message)
        if price_patterns:
            prices = [int(p.replace(',', '')) for p in price_patterns]
            if len(prices) == 1:
                if 'under' in message_lower or 'below' in message_lower:
                    preferences['max_budget'] = prices[0]
                elif 'over' in message_lower or 'above' in message_lower:
                    preferences['min_budget'] = prices[0]
                else:
                    preferences['target_budget'] = prices[0]
            elif len(prices) >= 2:
                preferences['min_budget'] = min(prices)
                preferences['max_budget'] = max(prices)
        
        # Budget-friendly keywords
        budget_keywords = ['budget', 'budget-friendly', 'affordable', 'cheap', 'economical', 'value']
        if any(keyword in message_lower for keyword in budget_keywords):
            preferences['priority_value'] = True
            # Set reasonable budget cap for budget-conscious users
            if 'max_budget' not in preferences:
                preferences['max_budget'] = 35000  # Budget-friendly cap
        
        # Feature preferences
        if any(term in message_lower for term in ['performance', 'fast', 'powerful', 'sporty']):
            preferences['priority_performance'] = True
        if any(term in message_lower for term in ['fuel efficient', 'mpg', 'economical']):
            preferences['priority_efficiency'] = True
        if any(term in message_lower for term in ['luxury', 'premium', 'high-end']):
            preferences['priority_luxury'] = True
        if any(term in message_lower for term in ['reliable', 'dependable', 'quality']):
            preferences['priority_reliability'] = True
        if any(term in message_lower for term in ['family', 'kids', 'children', 'space']):
            preferences['priority_family'] = True
        
        return preferences
    
    def _build_user_preference_profile(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive user preference profile"""
        preferences = user_inputs.get("extracted_preferences", {})
        input_category = user_inputs.get("input_category", "general_consultation")
        
        # Build preference scores (0-10 scale)
        preference_scores = {
            "performance": 5.0,
            "fuel_economy": 5.0, 
            "luxury": 5.0,
            "reliability": 7.0,  # Default high importance
            "value": 6.0,
            "safety": 7.0,      # Default high importance
            "technology": 5.0,
            "practicality": 5.0,
            "environmental": 5.0,
            "style": 5.0
        }
        
        # Adjust scores based on detected preferences
        if preferences.get("priority_performance"):
            preference_scores["performance"] = 9.0
        if preferences.get("priority_efficiency"):
            preference_scores["fuel_economy"] = 9.0
        if preferences.get("priority_luxury"):
            preference_scores["luxury"] = 9.0
        if preferences.get("priority_reliability"):
            preference_scores["reliability"] = 9.0
        if preferences.get("priority_family"):
            preference_scores["practicality"] = 9.0
            preference_scores["safety"] = 9.0
        
        # Environmental preference based on fuel type
        if preferences.get("fuel_type") in ["Electric", "Hybrid"]:
            preference_scores["environmental"] = 9.0
        
        # User classification
        user_type = self._classify_user_type(preferences, preference_scores)
        
        # Budget analysis
        budget_analysis = self._analyze_budget_preferences(preferences)
        
        return {
            "user_type": user_type,
            "input_category": input_category,
            "preference_scores": preference_scores,
            "vehicle_preferences": {
                "vehicle_type": preferences.get("vehicle_type"),
                "preferred_brand": preferences.get("preferred_brand"),
                "fuel_type": preferences.get("fuel_type")
            },
            "budget_analysis": budget_analysis,
            "priority_factors": self._identify_priority_factors(preference_scores),
            "consultation_complexity": self._assess_consultation_complexity(user_inputs),
            "recommendation_strategy": self._determine_recommendation_strategy(user_type, input_category)
        }
    
    def _classify_user_type(self, preferences: Dict[str, Any], scores: Dict[str, float]) -> str:
        """Classify user type based on preferences"""
        if preferences.get("priority_performance") or scores.get("performance", 0) > 8:
            return "Performance Enthusiast"
        elif preferences.get("priority_family") or scores.get("practicality", 0) > 8:
            return "Family-Focused"
        elif preferences.get("fuel_type") in ["Electric", "Hybrid"] or scores.get("environmental", 0) > 8:
            return "Eco-Conscious"
        elif preferences.get("priority_luxury") or scores.get("luxury", 0) > 8:
            return "Luxury-Oriented"
        elif scores.get("value", 0) > 7 and scores.get("reliability", 0) > 7:
            return "Value-Conscious"
        else:
            return "General Consumer"
    
    def _analyze_budget_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze budget-related preferences"""
        budget_info = {
            "has_specific_budget": False,
            "budget_range": "Not specified",
            "budget_category": "Unknown",
            "financing_likely": True
        }
        
        if "target_budget" in preferences:
            budget_info["has_specific_budget"] = True
            budget_info["budget_range"] = f"Around ${preferences['target_budget']:,}"
            budget_info["budget_category"] = self._categorize_budget(preferences["target_budget"])
        elif "min_budget" in preferences and "max_budget" in preferences:
            budget_info["has_specific_budget"] = True
            budget_info["budget_range"] = f"${preferences['min_budget']:,} - ${preferences['max_budget']:,}"
            avg_budget = (preferences["min_budget"] + preferences["max_budget"]) / 2
            budget_info["budget_category"] = self._categorize_budget(avg_budget)
        elif "max_budget" in preferences:
            budget_info["has_specific_budget"] = True
            budget_info["budget_range"] = f"Under ${preferences['max_budget']:,}"
            budget_info["budget_category"] = self._categorize_budget(preferences["max_budget"])
        elif "min_budget" in preferences:
            budget_info["has_specific_budget"] = True
            budget_info["budget_range"] = f"Over ${preferences['min_budget']:,}"
            budget_info["budget_category"] = self._categorize_budget(preferences["min_budget"])
        
        return budget_info
    
    def _categorize_budget(self, amount: float) -> str:
        """Categorize budget amount"""
        if amount < 25000:
            return "Economy"
        elif amount < 45000:
            return "Mid-range"
        elif amount < 80000:
            return "Premium"
        elif amount < 150000:
            return "Luxury"
        else:
            return "Ultra-luxury"
    
    def _identify_priority_factors(self, scores: Dict[str, float]) -> List[str]:
        """Identify top priority factors"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [factor for factor, score in sorted_scores[:3] if score > 6.0]
    
    def _assess_consultation_complexity(self, user_inputs: Dict[str, Any]) -> str:
        """Assess how complex the consultation needs to be"""
        preferences = user_inputs.get("extracted_preferences", {})
        input_category = user_inputs.get("input_category", "")
        
        complexity_factors = 0
        
        # Add complexity for specific requirements
        if len(preferences) > 3:
            complexity_factors += 1
        if input_category in ["vehicle_comparison", "dealer_connection"]:
            complexity_factors += 1
        if any(key in preferences for key in ["min_budget", "max_budget"]):
            complexity_factors += 1
        
        if complexity_factors >= 2:
            return "High"
        elif complexity_factors == 1:
            return "Medium"
        else:
            return "Low"
    
    def _determine_recommendation_strategy(self, user_type: str, input_category: str) -> str:
        """Determine the best recommendation strategy"""
        if input_category == "vehicle_recommendation":
            return "comprehensive_recommendation"
        elif input_category == "vehicle_comparison":
            return "comparative_analysis"
        elif input_category == "dealer_connection":
            return "dealer_focused"
        elif input_category == "pricing_inquiry":
            return "value_focused"
        else:
            return "consultative_guidance"
    
    def _prepare_database_query_params(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for database queries"""
        vehicle_prefs = user_profile.get("vehicle_preferences", {})
        budget_analysis = user_profile.get("budget_analysis", {})
        preference_scores = user_profile.get("preference_scores", {})
        
        query_params = {
            "search_criteria": {},
            "query_types": [],
            "result_limits": {}
        }
        
        # Vehicle search criteria
        search_criteria = {}
        if vehicle_prefs.get("vehicle_type"):
            search_criteria["body_type"] = vehicle_prefs["vehicle_type"]
        if vehicle_prefs.get("preferred_brand"):
            search_criteria["brand"] = vehicle_prefs["preferred_brand"]
        if vehicle_prefs.get("fuel_type"):
            search_criteria["fuel_type"] = vehicle_prefs["fuel_type"]
        
        # Budget constraints
        if "min_budget" in budget_analysis:
            search_criteria["price_min"] = budget_analysis["min_budget"]
        if "max_budget" in budget_analysis:
            search_criteria["price_max"] = budget_analysis["max_budget"]
        
        query_params["search_criteria"] = search_criteria
        
        # Determine what types of queries to run
        query_types = ["search_vehicles"]
        
        if user_profile.get("input_category") == "dealer_connection":
            query_types.append("find_dealers")
        if preference_scores.get("value", 0) > 7:
            query_types.append("analyze_market")
        if user_profile.get("consultation_complexity") == "High":
            query_types.append("browse_inventory")
        
        query_params["query_types"] = query_types
        
        # Result limits
        query_params["result_limits"] = {
            "vehicles": 8,
            "dealers": 5,
            "inventory": 10
        }
        
        return query_params
    
    def _coordinate_with_database_worker(self, state: MessageState, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate with CarDatabaseWorker to execute database queries"""
        try:
            # Import CarDatabaseWorker to use its public interface
            from arklex.env.workers.car_database_worker import CarDatabaseWorker
            
            # Use the public interface to execute coordinated query
            results = CarDatabaseWorker.execute_coordinated_query(query_params)
            
            logger.info(f"CarDatabaseWorker coordination completed: {results.get('total_results', 0)} total results")
            return results
            
        except Exception as e:
            logger.error(f"Error coordinating with CarDatabaseWorker: {str(e)}")
            # Return fallback results
            return {
                "vehicles": [],
                "dealers": [],
                "inventory": [],
                "market_analysis": {},
                "query_success": False,
                "error": str(e),
                "total_results": 0
            }
    
    def _create_personalized_recommendations(self, user_profile: Dict[str, Any], 
                                           database_results: Dict[str, Any], 
                                           user_inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create personalized recommendations by combining profile with database results"""
        vehicles = database_results.get("vehicles", [])
        preference_scores = user_profile.get("preference_scores", {})
        user_type = user_profile.get("user_type", "General Consumer")
        
        # Extract urban driving preference
        extracted_preferences = user_inputs.get("extracted_preferences", {})
        priority_value = extracted_preferences.get("priority_value", False)
        preferred_brand = extracted_preferences.get("preferred_brand")
        max_budget = extracted_preferences.get("max_budget")
        
        recommendations = []
        
        for vehicle in vehicles:
            # Apply budget filter if budget preference specified
            budget_suitability = self._calculate_budget_suitability(vehicle, priority_value, max_budget)
            
            # Apply brand preference boost
            brand_boost = 1.2 if preferred_brand and vehicle.get('brand', '').lower() == preferred_brand.lower() else 1.0
            
            # Calculate personalized match score
            base_match_score = self._calculate_personalized_match_score(vehicle, preference_scores)
            
            # Apply adjustments for city driving and budget
            adjusted_match_score = base_match_score * budget_suitability * brand_boost
            
            # Generate personalized insights
            insights = self._generate_vehicle_insights(vehicle, user_profile)
            
            recommendation = {
                "vehicle": vehicle,
                "match_score": adjusted_match_score,
                "match_percentage": round(adjusted_match_score * 10, 1),
                "personalized_insights": insights,
                "fit_analysis": self._analyze_fit_for_user(vehicle, user_profile),
                "pros_for_user": self._identify_pros_for_user(vehicle, user_profile, extracted_preferences),
                "considerations": self._identify_considerations_for_user(vehicle, user_profile, extracted_preferences),
                "budget_suitability": budget_suitability
            }
            
            recommendations.append(recommendation)
        
        # Sort by adjusted match score
        recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        
        return recommendations
    
    def _calculate_budget_suitability(self, vehicle: Dict[str, Any], priority_value: bool, max_budget: Optional[int]) -> float:
        """Calculate budget suitability based on user's value priorities"""
        vehicle_price = vehicle.get('msrp', 0)
        
        # If user specified max budget, apply hard filter
        if max_budget and vehicle_price > max_budget:
            return 0.6  # Significantly reduce score for over-budget vehicles
        
        # If user prioritizes value/budget-friendly options
        if priority_value:
            if vehicle_price < 25000:
                return 1.2  # Boost for very affordable
            elif vehicle_price < 35000:
                return 1.1  # Boost for affordable
            elif vehicle_price > 50000:
                return 0.8  # Reduce for expensive
        
        return 1.0  # Neutral if no budget priority
    
    def _calculate_personalized_match_score(self, vehicle: Dict[str, Any], 
                                          preference_scores: Dict[str, float]) -> float:
        """Calculate how well vehicle matches user's specific preferences"""
        total_score = 0.0
        total_weight = 0.0
        
        # Map vehicle characteristics to preference dimensions
        vehicle_scores = {
            "performance": min(10.0, vehicle.get("horsepower", 200) / 30),
            "fuel_economy": min(10.0, (vehicle.get("mpg_city", 25) + vehicle.get("mpg_highway", 35)) / 5),
            "luxury": 7.0 if vehicle.get("price", 0) > 40000 else 5.0,
            "reliability": 7.0,  # Would come from reliability database
            "value": max(1.0, 10.0 - (vehicle.get("price", 30000) / 5000)),
            "safety": 8.0,      # Would come from safety ratings
            "technology": 6.0,   # Would be determined by features
            "practicality": 8.0 if vehicle.get("body_type") in ["SUV", "Sedan"] else 6.0,
            "environmental": 10.0 if vehicle.get("fuel_type") == "Electric" else 8.0 if vehicle.get("fuel_type") == "Hybrid" else 4.0,
            "style": 7.0        # Subjective, could be enhanced with style ratings
        }
        
        # Calculate weighted score based on user preferences
        for pref, user_importance in preference_scores.items():
            if pref in vehicle_scores:
                weight = user_importance / 10.0
                score = vehicle_scores[pref] * weight
                total_score += score
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 5.0
    
    def _generate_vehicle_insights(self, vehicle: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized insights about why this vehicle fits the user"""
        user_type = user_profile.get("user_type", "General Consumer")
        priority_factors = user_profile.get("priority_factors", [])
        
        insights = {
            "why_recommended": f"This {vehicle.get('brand', '')} {vehicle.get('model', '')} aligns well with your {user_type.lower()} profile",
            "key_strengths": [],
            "user_specific_benefits": []
        }
        
        # Add user-specific benefits based on their type
        if user_type == "Performance Enthusiast":
            insights["user_specific_benefits"].append("Strong acceleration and responsive handling")
        elif user_type == "Family-Focused":
            insights["user_specific_benefits"].append("Excellent safety ratings and spacious interior")
        elif user_type == "Eco-Conscious":
            insights["user_specific_benefits"].append("Outstanding fuel efficiency and low emissions")
        elif user_type == "Luxury-Oriented":
            insights["user_specific_benefits"].append("Premium materials and advanced features")
        elif user_type == "Value-Conscious":
            insights["user_specific_benefits"].append("Excellent value with low ownership costs")
        
        return insights
    
    def _analyze_fit_for_user(self, vehicle: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well the vehicle fits the user's specific needs"""
        fit_analysis = {
            "overall_fit": "Good",
            "fit_score": 7.5,
            "lifestyle_match": "Suitable",
            "budget_fit": "Within range"
        }
        
        # Analyze budget fit
        vehicle_price = vehicle.get("price", 0)
        budget_analysis = user_profile.get("budget_analysis", {})
        
        if budget_analysis.get("has_specific_budget"):
            if "max_budget" in budget_analysis and vehicle_price > budget_analysis["max_budget"]:
                fit_analysis["budget_fit"] = "Above budget"
                fit_analysis["fit_score"] -= 2.0
            elif "min_budget" in budget_analysis and vehicle_price < budget_analysis["min_budget"]:
                fit_analysis["budget_fit"] = "Below budget range"
                fit_analysis["fit_score"] -= 1.0
            else:
                fit_analysis["budget_fit"] = "Perfect budget match"
                fit_analysis["fit_score"] += 1.0
        
        # Update overall fit based on score
        if fit_analysis["fit_score"] >= 8.5:
            fit_analysis["overall_fit"] = "Excellent"
        elif fit_analysis["fit_score"] >= 7.0:
            fit_analysis["overall_fit"] = "Good"
        else:
            fit_analysis["overall_fit"] = "Fair"
        
        return fit_analysis
    
    def _identify_pros_for_user(self, vehicle: Dict[str, Any], user_profile: Dict[str, Any], extracted_preferences: Dict[str, Any]) -> List[str]:
        """Identify specific pros for this user"""
        pros = []
        preference_scores = user_profile.get("preference_scores", {})
        
        # Fuel economy
        mpg_combined = (vehicle.get("mpg_city", 25) + vehicle.get("mpg_highway", 35)) / 2
        if mpg_combined > 35 and preference_scores.get("fuel_economy", 0) > 6:
            pros.append(f"Excellent fuel economy ({mpg_combined:.1f} MPG combined)")
        
        # Value proposition
        if vehicle.get("price", 0) < 35000 and preference_scores.get("value", 0) > 6:
            pros.append("Great value for money")
        
        # Environmental benefits
        if vehicle.get("fuel_type") in ["Electric", "Hybrid"] and preference_scores.get("environmental", 0) > 6:
            pros.append("Environmentally friendly powertrain")
        
        # Brand preference
        if extracted_preferences.get("preferred_brand") and vehicle.get('brand', '').lower() == extracted_preferences["preferred_brand"].lower():
            pros.append(f"Preferred brand: {extracted_preferences['preferred_brand']}")
        
        return pros[:3]  # Limit to top 3 pros
    
    def _identify_considerations_for_user(self, vehicle: Dict[str, Any], user_profile: Dict[str, Any], extracted_preferences: Dict[str, Any]) -> List[str]:
        """Identify potential considerations for this user"""
        considerations = []
        preference_scores = user_profile.get("preference_scores", {})
        
        # Performance considerations
        if preference_scores.get("performance", 0) > 7 and vehicle.get("horsepower", 200) < 250:
            considerations.append("May lack the performance you're seeking")
        
        # Luxury considerations
        if preference_scores.get("luxury", 0) > 7 and vehicle.get("price", 0) < 40000:
            considerations.append("Limited luxury features at this price point")
        
        # Brand consideration
        if extracted_preferences.get("preferred_brand") and vehicle.get('brand', '').lower() != extracted_preferences["preferred_brand"].lower():
            considerations.append(f"Not a preferred brand: {vehicle.get('brand', '')}")
        
        return considerations[:2]  # Limit to top 2 considerations
    
    def _generate_comprehensive_response(self, user_profile: Dict[str, Any], 
                                       database_results: Dict[str, Any],
                                       recommendations: List[Dict[str, Any]], 
                                       user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive consultation response"""
        return {
            "consultation_summary": {
                "user_type": user_profile.get("user_type", "General Consumer"),
                "consultation_type": user_profile.get("input_category", "general_consultation"),
                "complexity_level": user_profile.get("consultation_complexity", "Medium"),
                "timestamp": datetime.now().isoformat()
            },
            "preference_analysis": {
                "top_priorities": user_profile.get("priority_factors", []),
                "budget_analysis": user_profile.get("budget_analysis", {}),
                "vehicle_preferences": user_profile.get("vehicle_preferences", {})
            },
            "recommendations": recommendations,
            "database_insights": {
                "total_vehicles_found": len(database_results.get("vehicles", [])),
                "dealers_available": len(database_results.get("dealers", [])),
                "market_context": database_results.get("market_analysis", {})
            },
            "next_steps": self._generate_next_steps(user_profile, recommendations),
            "consultation_quality": self._assess_consultation_quality(user_profile, database_results, recommendations)
        }
    
    def _generate_next_steps(self, user_profile: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> List[str]:
        """Generate personalized next steps"""
        next_steps = []
        
        if recommendations:
            next_steps.append(f"Review the {len(recommendations)} personalized vehicle recommendations")
            next_steps.append("Schedule test drives for your top 2-3 choices")
        
        input_category = user_profile.get("input_category", "")
        if input_category == "dealer_connection":
            next_steps.append("Contact recommended dealers for inventory and pricing")
        elif input_category == "pricing_inquiry":
            next_steps.append("Compare pricing across multiple dealers")
        
        next_steps.append("Consider financing options and total cost of ownership")
        next_steps.append("Verify insurance costs for your preferred vehicles")
        
        return next_steps
    
    def _assess_consultation_quality(self, user_profile: Dict[str, Any], 
                                   database_results: Dict[str, Any], 
                                   recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of the consultation provided"""
        quality_score = 7.0  # Base score
        
        # Increase score for successful database queries
        if database_results.get("query_success"):
            quality_score += 1.0
        
        # Increase score for personalized recommendations
        if len(recommendations) > 0:
            quality_score += 1.0
            
        # Increase score for complex preference analysis
        if user_profile.get("consultation_complexity") == "High":
            quality_score += 0.5
        
        return {
            "quality_score": min(10.0, quality_score),
            "completeness": "High" if quality_score >= 8.0 else "Medium" if quality_score >= 6.0 else "Low",
            "personalization_level": "High",
            "user_satisfaction_prediction": "High" if quality_score >= 8.0 else "Medium"
        }
    
    def _format_user_response(self, comprehensive_response: Dict[str, Any]) -> str:
        """Format the comprehensive response for user consumption"""
        consultation_summary = comprehensive_response.get("consultation_summary", {})
        preference_analysis = comprehensive_response.get("preference_analysis", {})
        recommendations = comprehensive_response.get("recommendations", [])
        database_insights = comprehensive_response.get("database_insights", {})
        next_steps = comprehensive_response.get("next_steps", [])
        
        response_parts = []
        
        # Header
        response_parts.append(f"🚗 **Personalized Car Consultation for {consultation_summary.get('user_type', 'Valued Customer')}**\n")
        
        # Preference summary
        top_priorities = preference_analysis.get("top_priorities", [])
        if top_priorities:
            response_parts.append(f"**Your Top Priorities:** {', '.join(top_priorities)}")
        
        budget_info = preference_analysis.get("budget_analysis", {})
        if budget_info.get("has_specific_budget"):
            response_parts.append(f"**Budget Range:** {budget_info.get('budget_range', 'Not specified')}")
        
        response_parts.append("")  # Empty line
        
        # Recommendations
        if recommendations:
            response_parts.append(f"**🎯 Found {len(recommendations)} Personalized Vehicle Recommendations:**\n")
            
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                vehicle = rec["vehicle"]
                response_parts.append(f"**{i}. {vehicle.get('year', 'N/A')} {vehicle.get('brand', 'N/A')} {vehicle.get('model', 'N/A')}**")
                response_parts.append(f"   • Match Score: {rec.get('match_percentage', 0)}%")
                response_parts.append(f"   • Price: ${vehicle.get('price', 0):,}")
                response_parts.append(f"   • Type: {vehicle.get('body_type', 'N/A')} | Fuel: {vehicle.get('fuel_type', 'N/A')}")
                
                pros = rec.get("pros_for_user", [])
                if pros:
                    response_parts.append(f"   • Why it's great for you: {pros[0]}")
                
                response_parts.append("")  # Empty line
        
        # Database insights
        total_vehicles = database_insights.get("total_vehicles_found", 0)
        if total_vehicles > len(recommendations):
            response_parts.append(f"*Found {total_vehicles} total vehicles matching your criteria*")
        
        # Next steps
        if next_steps:
            response_parts.append("**🎯 Recommended Next Steps:**")
            for step in next_steps[:4]:  # Show top 4 steps
                response_parts.append(f"• {step}")
        
        response_parts.append("\n*I've analyzed your preferences and coordinated with our vehicle database to provide these personalized recommendations. Would you like more details about any specific vehicle or assistance with next steps?*")
        
        return "\n".join(response_parts)
    
    def _execute(self, msg_state: MessageState, **kwargs: Any) -> MessageState:
        """Execute the car advisor workflow with problem classification priority"""
        try:
            # Initialize slots early
            if msg_state.slots is None:
                msg_state.slots = {}
            if "car_advisor" not in msg_state.slots:
                msg_state.slots["car_advisor"] = []
            
            # Extract user message for problem classification
            user_message = ""
            if msg_state.user_message:
                user_message = msg_state.user_message.message if hasattr(msg_state.user_message, 'message') else str(msg_state.user_message)
            
            # PRIORITY 1: Classify user problem type
            problem_classification = self.classify_user_problem(user_message)
            logger.info(f"Problem classification: {problem_classification}")
            
            # Store problem classification in slots
            msg_state.slots["car_advisor"].append(
                Slot(
                    name="problem_classification",
                    type="str",
                    value=json.dumps(problem_classification, ensure_ascii=False),
                    description="User problem classification result"
                )
            )
            
            # PRIORITY 2: Handle high-priority problems immediately
            if problem_classification.get("priority") == "high" or \
               problem_classification.get("context", {}).get("requires_immediate_action"):
                logger.info(f"Handling high-priority problem: {problem_classification['type']}")
                
                # Generate problem-focused response
                problem_response = self.generate_problem_focused_response(problem_classification, user_message)
                
                msg_state.response = problem_response
                msg_state.status = StatusEnum.COMPLETE
                
                # Store that this was a problem-focused interaction
                msg_state.slots["car_advisor"].append(
                    Slot(
                        name="interaction_type",
                        type="str",
                        value="problem_resolution",
                        description="Type of interaction provided"
                    )
                )
                
                return msg_state
            
            # PRIORITY 3: For recommendation/general cases, proceed with full workflow
            elif problem_classification.get("context", {}).get("needs_preference_collection"):
                logger.info(f"Proceeding with preference collection for: {problem_classification['type']}")
                
                # Store interaction type
                msg_state.slots["car_advisor"].append(
                    Slot(
                        name="interaction_type",
                        type="str",
                        value="preference_based_consultation",
                        description="Type of interaction provided"
                    )
                )
                
                # Compile and run the coordination graph
                graph = self.action_graph.compile()
                result = graph.invoke(msg_state)
                
                return result
            
            # PRIORITY 4: Handle specific queries 
            else:
                logger.info(f"Handling specific query: {problem_classification['type']}")
                
                # Generate targeted response for specific queries
                specific_response = self.generate_problem_focused_response(problem_classification, user_message)
                
                msg_state.response = specific_response
                msg_state.status = StatusEnum.COMPLETE
                
                msg_state.slots["car_advisor"].append(
                    Slot(
                        name="interaction_type",
                        type="str",
                        value="specific_query_resolution",
                        description="Type of interaction provided"
                    )
                )
                
                return msg_state
            
        except Exception as e:
            logger.error(f"Error in car advisor execution: {str(e)}")
            msg_state.response = f"I encountered an error while analyzing your automotive needs: {str(e)}. Please try rephrasing your question or providing more specific details about what you're looking for."
            msg_state.status = StatusEnum.INCOMPLETE
            return msg_state