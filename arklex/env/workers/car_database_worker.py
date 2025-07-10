"""
Car Database Worker - Specialized worker for automotive database operations

This worker handles car-specific database operations including vehicle search,
dealer management, inventory browsing, order tracking, and user preference management.
"""

import logging
import sqlite3
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.env.prompts import load_prompts
from arklex.env.tools.utils import ToolGenerator
from arklex.utils.utils import chunk_string
from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.utils.model_config import MODEL

logger = logging.getLogger(__name__)


@register_worker
class CarDatabaseWorker(BaseWorker):
    """Specialized worker for automotive database operations and queries"""
    
    description: str = "Handle comprehensive car database operations including vehicle search, dealer management, inventory browsing, order tracking, and user preference management for automotive advisory systems."

    def __init__(self) -> None:
        self.llm: BaseChatModel = ChatOpenAI(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        self.actions: Dict[str, str] = {
            "SearchVehicles": "Search for vehicles based on criteria",
            "BrowseInventory": "Browse dealer inventory and availability", 
            "FindDealers": "Find and locate nearby dealerships",
            "TrackOrder": "Track vehicle order status and updates",
            "ManagePreferences": "Manage user preferences and profiles",
            "AnalyzeMarket": "Analyze market trends and pricing",
            "ExecuteQuery": "Execute coordinated database query with multiple operations",
            "Others": "Other car-related database operations"
        }
        self.db_path: str = "examples/car_advisor/car_advisor_db.sqlite"
        self.action_graph: StateGraph = self._create_action_graph()

    def search_vehicles(self, state: MessageState) -> MessageState:
        """Search for vehicles based on user criteria"""
        try:
            # Correctly extract message content from ConvoMessage object
            user_message = ""
            if state.user_message:
                user_message = state.user_message.message if hasattr(state.user_message, 'message') else str(state.user_message)
            
            criteria = self._extract_search_criteria(user_message)
            
            results = self._query_vehicles(criteria)
            
            if results:
                response = self._format_vehicle_results(results)
                state.response = f"Found {len(results)} vehicles matching your criteria:\n{response}"
                state.status = StatusEnum.COMPLETE
            else:
                state.response = "No vehicles found matching your criteria. Our database contains 207 vehicles. Please try adjusting your search parameters or let me know what specific type of vehicle you're looking for."
                state.status = StatusEnum.INCOMPLETE
                
        except Exception as e:
            logger.error(f"Error in search_vehicles: {e}")
            state.response = f"I encountered an error while searching for vehicles: {str(e)}. Please try again."
            state.status = StatusEnum.INCOMPLETE
            
        return state

    def browse_inventory(self, state: MessageState) -> MessageState:
        """Browse dealer inventory and availability"""
        try:
            inventory_data = self._get_inventory_data()
            
            if inventory_data:
                response = self._format_inventory_results(inventory_data)
                state.response = f"Current inventory information:\n{response}"
                state.status = StatusEnum.COMPLETE
            else:
                state.response = "No inventory information available at the moment. Please check back later."
                state.status = StatusEnum.INCOMPLETE
                
        except Exception as e:
            logger.error(f"Error in browse_inventory: {e}")
            state.response = f"I encountered an error while browsing inventory: {str(e)}. Please try again."
            state.status = StatusEnum.INCOMPLETE
            
        return state

    def find_dealers(self, state: MessageState) -> MessageState:
        """Find and locate nearby dealerships"""
        try:
            # Correctly extract message content from ConvoMessage object
            user_message = ""
            if state.user_message:
                user_message = state.user_message.message if hasattr(state.user_message, 'message') else str(state.user_message)
            
            location = self._extract_location(user_message)
            dealers = self._query_dealers(location)
            
            if dealers:
                response = self._format_dealer_results(dealers)
                state.response = f"Found {len(dealers)} dealers{' near ' + location if location else ''}:\n{response}"
                state.status = StatusEnum.COMPLETE
            else:
                state.response = f"No dealers found{' near ' + location if location else ''}. We have 5 dealers in our database. Please try a different location or let me show you all available dealers."
                state.status = StatusEnum.INCOMPLETE
                
        except Exception as e:
            logger.error(f"Error in find_dealers: {e}")
            state.response = f"I encountered an error while finding dealers: {str(e)}. Please try again."
            state.status = StatusEnum.INCOMPLETE
            
        return state

    def track_order(self, state: MessageState) -> MessageState:
        """Track vehicle order status and updates"""
        try:
            # Correctly extract message content from ConvoMessage object
            user_message = ""
            if state.user_message:
                user_message = state.user_message.message if hasattr(state.user_message, 'message') else str(state.user_message)
            
            order_id = self._extract_order_id(user_message)
            
            if order_id:
                order_status = self._query_order_status(order_id)
                if order_status:
                    response = self._format_order_status(order_status)
                    state.response = f"Order Status for {order_id}:\n{response}"
                    state.status = StatusEnum.COMPLETE
                else:
                    state.response = f"Order {order_id} not found in our system. Please verify the order ID and try again."
                    state.status = StatusEnum.INCOMPLETE
            else:
                state.response = "Please provide a valid order ID to track your order. Order IDs typically consist of numbers and letters."
                state.status = StatusEnum.INCOMPLETE
                
        except Exception as e:
            logger.error(f"Error in track_order: {e}")
            state.response = f"I encountered an error while tracking your order: {str(e)}. Please try again."
            state.status = StatusEnum.INCOMPLETE
            
        return state

    def manage_preferences(self, state: MessageState) -> MessageState:
        """Manage user preferences and profiles"""
        try:
            # Correctly extract message content from ConvoMessage object
            user_message = ""
            if state.user_message:
                user_message = state.user_message.message if hasattr(state.user_message, 'message') else str(state.user_message)
            
            preferences = self._extract_preferences(user_message)
            success = self._update_user_preferences(preferences)
            
            if success:
                state.response = "Your preferences have been updated successfully. I'll use these preferences to provide better recommendations."
                state.status = StatusEnum.COMPLETE
            else:
                state.response = "Failed to update preferences. Please try again or provide more specific preference information."
                state.status = StatusEnum.INCOMPLETE
                
        except Exception as e:
            logger.error(f"Error in manage_preferences: {e}")
            state.response = f"I encountered an error while managing your preferences: {str(e)}. Please try again."
            state.status = StatusEnum.INCOMPLETE
            
        return state

    def analyze_market(self, state: MessageState) -> MessageState:
        """Analyze market trends and pricing"""
        try:
            market_data = self._get_market_analysis()
            
            if market_data:
                response = self._format_market_analysis(market_data)
                state.response = f"Market Analysis:\n{response}"
                state.status = StatusEnum.COMPLETE
            else:
                state.response = "Market analysis data is not available at the moment. Please check back later."
                state.status = StatusEnum.INCOMPLETE
                
        except Exception as e:
            logger.error(f"Error in analyze_market: {e}")
            state.response = f"I encountered an error while analyzing market data: {str(e)}. Please try again."
            state.status = StatusEnum.INCOMPLETE
            
        return state

    def execute_query(self, state: MessageState) -> MessageState:
        """Execute coordinated database query with multiple operations for CarAdvisorWorker"""
        try:
            # Extract query parameters from orchestrator message or slots
            query_params = self._extract_query_parameters(state)
            
            # Execute multiple database operations as requested
            results = {
                "vehicles": [],
                "dealers": [],
                "inventory": [],
                "market_analysis": {},
                "query_success": True,
                "total_results": 0
            }
            
            query_types = query_params.get("query_types", ["search_vehicles"])
            search_criteria = query_params.get("search_criteria", {})
            result_limits = query_params.get("result_limits", {})
            
            # Execute vehicle search if requested
            if "search_vehicles" in query_types:
                vehicles = self._query_vehicles(search_criteria)
                results["vehicles"] = vehicles[:result_limits.get("vehicles", 10)]
                results["total_results"] += len(vehicles)
                logger.info(f"Found {len(vehicles)} vehicles matching criteria")
            
            # Execute dealer search if requested  
            if "find_dealers" in query_types:
                location = search_criteria.get("location", "")
                dealers = self._query_dealers(location)
                results["dealers"] = dealers[:result_limits.get("dealers", 5)]
                logger.info(f"Found {len(dealers)} dealers")
            
            # Execute inventory search if requested
            if "browse_inventory" in query_types:
                inventory = self._get_inventory_data()
                results["inventory"] = inventory[:result_limits.get("inventory", 10)]
                logger.info(f"Found {len(inventory)} inventory items")
            
            # Execute market analysis if requested
            if "analyze_market" in query_types:
                market_data = self._get_market_analysis()
                if market_data:
                    results["market_analysis"] = market_data
                    logger.info("Market analysis completed")
            
            # Format comprehensive response
            response_text = self._format_coordinated_results(results, query_params)
            state.response = response_text
            state.status = StatusEnum.COMPLETE
            
            # Store results in slots for CarAdvisorWorker coordination
            if state.slots is None:
                state.slots = {}
            if "database_results" not in state.slots:
                state.slots["database_results"] = []
            
            from arklex.utils.slot import Slot
            state.slots["database_results"] = [
                Slot(
                    name="coordinated_query_results",
                    type="str",
                    value=json.dumps(results, ensure_ascii=False),
                    description="Results from coordinated database operations"
                )
            ]
            
            logger.info(f"Coordinated database query completed successfully with {results['total_results']} total results")
            return state
            
        except Exception as e:
            logger.error(f"Error in execute_query: {e}")
            state.response = f"Database query failed: {str(e)}. Please try again with simpler criteria."
            state.status = StatusEnum.INCOMPLETE
            return state

    def verify_action(self, msg_state: MessageState) -> str:
        """Determine which car database action to take based on user intent"""
        user_intent: str = msg_state.orchestrator_message.attribute.get("task", "")
        actions_info: str = "\n".join(
            [f"{name}: {description}" for name, description in self.actions.items()]
        )
        actions_name: str = ", ".join(self.actions.keys())

        prompts: Dict[str, str] = load_prompts(msg_state.bot_config)
        prompt: PromptTemplate = PromptTemplate.from_template(
            prompts.get("car_database_action_prompt", 
                       "Based on the user intent: {user_intent}\n\n"
                       "Available actions:\n{actions_info}\n\n"
                       "Choose the most appropriate action from: {actions_name}")
        )
        
        input_prompt = prompt.invoke({
            "user_intent": user_intent,
            "actions_info": actions_info,
            "actions_name": actions_name,
        })
        
        chunked_prompt: str = chunk_string(
            input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"]
        )
        
        logger.info(f"Chunked prompt for car database action: {chunked_prompt}")
        
        final_chain = self.llm | StrOutputParser()
        try:
            answer: str = final_chain.invoke(chunked_prompt)
            for action_name in self.actions.keys():
                if action_name in answer:
                    logger.info(f"Chosen car database action: {action_name}")
                    return action_name
            logger.info("Default car database action chosen: Others")
            return "Others"
        except Exception as e:
            logger.error(f"Error choosing car database action: {e}")
            return "Others"

    def _create_action_graph(self) -> StateGraph:
        """Create the action workflow graph"""
        workflow: StateGraph = StateGraph(MessageState)
        
        workflow.add_node("SearchVehicles", self.search_vehicles)
        workflow.add_node("BrowseInventory", self.browse_inventory)
        workflow.add_node("FindDealers", self.find_dealers)
        workflow.add_node("TrackOrder", self.track_order)
        workflow.add_node("ManagePreferences", self.manage_preferences)
        workflow.add_node("AnalyzeMarket", self.analyze_market)
        workflow.add_node("ExecuteQuery", self.execute_query)
        workflow.add_node("Others", ToolGenerator.generate)
        workflow.add_node("tool_generator", ToolGenerator.context_generate)
        
        workflow.add_conditional_edges(START, self.verify_action)
        
        for action in ["SearchVehicles", "BrowseInventory", "FindDealers", 
                      "TrackOrder", "ManagePreferences", "AnalyzeMarket", "ExecuteQuery"]:
            workflow.add_edge(action, "tool_generator")
            
        return workflow

    def _execute(self, msg_state: MessageState, **kwargs) -> MessageState:
        """Execute the car database worker workflow"""
        try:
            self._ensure_database_exists()
            graph = self.action_graph.compile()
            result: MessageState = graph.invoke(msg_state)
            return result
            
        except Exception as e:
            logger.error(f"Error executing car database worker: {e}")
            msg_state.response = f"I encountered an error while processing your request: {str(e)}. Please try again."
            msg_state.status = StatusEnum.INCOMPLETE
            return msg_state

    def _ensure_database_exists(self):
        """Ensure the car database exists and is accessible"""
        if not Path(self.db_path).exists():
            logger.error(f"Car database not found at {self.db_path}")
            raise FileNotFoundError(f"Car database not found at {self.db_path}")

    def _get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def _extract_search_criteria(self, message: str) -> Dict[str, Any]:
        """Extract vehicle search criteria from user message"""
        criteria = {}
        message_lower = message.lower()
        
        # Extract brand
        brands = ['toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi', 'chevrolet', 'nissan', 'hyundai', 'kia', 'mazda', 'subaru', 'volkswagen', 'tesla', 'lexus', 'acura', 'infiniti', 'cadillac', 'lincoln', 'buick']
        for brand in brands:
            if brand in message_lower:
                criteria['brand'] = brand.title()
                break
                
        # Extract body type
        body_types = ['sedan', 'suv', 'hatchback', 'coupe', 'convertible', 'wagon', 'pickup', 'truck']
        for body_type in body_types:
            if body_type in message_lower:
                criteria['body_type'] = body_type.title()
                break
                
        # Extract fuel type
        if 'electric' in message_lower or 'ev' in message_lower:
            criteria['fuel_type'] = 'Electric'
        elif 'hybrid' in message_lower:
            criteria['fuel_type'] = 'Hybrid'
        elif 'gas' in message_lower or 'gasoline' in message_lower:
            criteria['fuel_type'] = 'Gasoline'
            
        # Extract price range
        price_pattern = r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        prices = re.findall(price_pattern, message)
        if len(prices) >= 2:
            criteria['price_min'] = float(prices[0].replace(',', ''))
            criteria['price_max'] = float(prices[1].replace(',', ''))
        elif len(prices) == 1:
            if 'under' in message_lower or 'below' in message_lower:
                criteria['price_max'] = float(prices[0].replace(',', ''))
            elif 'over' in message_lower or 'above' in message_lower:
                criteria['price_min'] = float(prices[0].replace(',', ''))
                
        # Extract year
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, message)
        if years:
            criteria['year'] = int(years[0])
            
        return criteria

    def _query_vehicles(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query vehicles from database based on criteria"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            where_clauses = []
            params = []
            
            if 'brand' in criteria:
                where_clauses.append("LOWER(brand) = LOWER(?)")
                params.append(criteria['brand'])
                
            if 'body_type' in criteria:
                where_clauses.append("LOWER(body_type) = LOWER(?)")
                params.append(criteria['body_type'])
                
            if 'fuel_type' in criteria:
                where_clauses.append("LOWER(fuel_type) = LOWER(?)")
                params.append(criteria['fuel_type'])
                
            if 'price_min' in criteria:
                where_clauses.append("msrp >= ?")
                params.append(criteria['price_min'])
                
            if 'price_max' in criteria:
                where_clauses.append("msrp <= ?")
                params.append(criteria['price_max'])
                
            if 'year' in criteria:
                where_clauses.append("year = ?")
                params.append(criteria['year'])
                
            query = "SELECT * FROM vehicles"
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            query += " LIMIT 10"
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                vehicle = dict(zip(columns, row))
                results.append(vehicle)
                
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error querying vehicles: {e}")
            return []

    def _format_vehicle_results(self, results: List[Dict[str, Any]]) -> str:
        """Format vehicle search results for display"""
        if not results:
            return "No vehicles found."
            
        formatted_results = []
        for vehicle in results:
            result_text = f"• {vehicle['year']} {vehicle['brand']} {vehicle['model']} {vehicle.get('trim', '')}"
            result_text += f"\n  Price: ${vehicle.get('msrp', 'N/A'):,.0f}"
            result_text += f"\n  Type: {vehicle.get('body_type', 'N/A')}"
            result_text += f"\n  Fuel: {vehicle.get('fuel_type', 'N/A')}"
            result_text += f"\n  MPG: {vehicle.get('mpg_city', 'N/A')}/{vehicle.get('mpg_highway', 'N/A')} (city/highway)"
            if vehicle.get('horsepower'):
                result_text += f"\n  Power: {vehicle['horsepower']} HP"
            result_text += "\n"
            formatted_results.append(result_text)
            
        return "\n".join(formatted_results)

    def _get_inventory_data(self) -> List[Dict[str, Any]]:
        """Get current inventory data"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT i.*, v.brand, v.model, v.year, v.trim, v.msrp, v.body_type
                FROM inventory i
                JOIN vehicles v ON i.vehicle_id = v.id
                WHERE i.status = 'Available'
                LIMIT 20
            """)
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                item = dict(zip(columns, row))
                results.append(item)
                
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting inventory data: {e}")
            return []

    def _format_inventory_results(self, inventory: List[Dict[str, Any]]) -> str:
        """Format inventory results for display"""
        if not inventory:
            return "No inventory data available."
            
        formatted_results = []
        for item in inventory:
            result_text = f"• {item.get('year', 'N/A')} {item.get('brand', 'N/A')} {item.get('model', 'N/A')}"
            if item.get('trim'):
                result_text += f" {item['trim']}"
            result_text += f"\n  Status: {item.get('status', 'N/A')}"
            result_text += f"\n  Condition: {item.get('condition_type', 'N/A')}"
            result_text += f"\n  Price: ${item.get('asking_price', item.get('msrp', 0)):,.0f}"
            if item.get('exterior_color'):
                result_text += f"\n  Color: {item['exterior_color']}"
            if item.get('mileage') is not None:
                result_text += f"\n  Mileage: {item['mileage']:,} miles"
            if item.get('vin'):
                result_text += f"\n  VIN: {item['vin'][-6:]}"  # Show last 6 digits
            result_text += "\n"
            formatted_results.append(result_text)
            
        return "\n".join(formatted_results)

    def _extract_location(self, message: str) -> str:
        """Extract location from user message"""
        location_keywords = ['near', 'in', 'around', 'close to', 'by']
        message_lower = message.lower()
        
        for keyword in location_keywords:
            if keyword in message_lower:
                # Find text after the keyword
                parts = message_lower.split(keyword, 1)
                if len(parts) > 1:
                    location_part = parts[1].strip()
                    # Take first few words as location
                    location_words = location_part.split()[:3]
                    return ' '.join(location_words).strip(',.')
                    
        return ""

    def _query_dealers(self, location: str = "") -> List[Dict[str, Any]]:
        """Query dealers near location"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            if location:
                # Search by city or state
                cursor.execute("""
                    SELECT * FROM dealers 
                    WHERE LOWER(city) LIKE LOWER(?) OR LOWER(state) LIKE LOWER(?)
                    LIMIT 10
                """, (f"%{location}%", f"%{location}%"))
            else:
                # Return all dealers
                cursor.execute("SELECT * FROM dealers LIMIT 10")
                
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                dealer = dict(zip(columns, row))
                results.append(dealer)
                
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error querying dealers: {e}")
            return []

    def _format_dealer_results(self, dealers: List[Dict[str, Any]]) -> str:
        """Format dealer results for display"""
        if not dealers:
            return "No dealers found."
            
        formatted_results = []
        for dealer in dealers:
            result_text = f"• {dealer.get('name', 'N/A')}"
            result_text += f"\n  Location: {dealer.get('city', 'N/A')}, {dealer.get('state', 'N/A')}"
            if dealer.get('address_street'):
                result_text += f"\n  Address: {dealer['address_street']}"
            if dealer.get('phone'):
                result_text += f"\n  Phone: {dealer['phone']}"
            if dealer.get('customer_rating'):
                result_text += f"\n  Rating: {dealer['customer_rating']}/5.0"
            brands = dealer.get('brand_affiliations', '')
            if brands:
                result_text += f"\n  Brands: {brands}"
            result_text += "\n"
            formatted_results.append(result_text)
            
        return "\n".join(formatted_results)

    def _extract_order_id(self, message: str) -> Optional[str]:
        """Extract order ID from user message"""
        # Look for patterns like ORD123, ORDER-456, #789, etc.
        patterns = [
            r'(?:order\s*(?:id|number)?:?\s*)?([A-Z0-9\-]{6,})',
            r'#([A-Z0-9\-]{3,})',
            r'(?:id|number)\s*([A-Z0-9\-]{3,})'
        ]
        
        message_upper = message.upper()
        for pattern in patterns:
            match = re.search(pattern, message_upper)
            if match:
                return match.group(1)
        return None

    def _query_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Query order status from database"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT o.*, v.brand, v.model, v.year, v.trim
                FROM orders o
                LEFT JOIN inventory i ON o.inventory_id = i.id
                LEFT JOIN vehicles v ON i.vehicle_id = v.id
                WHERE UPPER(o.order_number) = UPPER(?)
            """, (order_id,))
            
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                result = dict(zip(columns, row))
                conn.close()
                return result
                
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error querying order status: {e}")
            return None

    def _format_order_status(self, order_data: Dict[str, Any]) -> str:
        """Format order status for display"""
        result = f"Order Number: {order_data.get('order_number', 'N/A')}\n"
        result += f"Vehicle: {order_data.get('year', 'N/A')} {order_data.get('brand', 'N/A')} {order_data.get('model', 'N/A')}"
        if order_data.get('trim'):
            result += f" {order_data['trim']}"
        result += f"\nStatus: {order_data.get('status', 'N/A')}"
        result += f"\nCreated: {order_data.get('created_at', 'N/A')}"
        if order_data.get('expected_delivery_date'):
            result += f"\nExpected Delivery: {order_data['expected_delivery_date']}"
        if order_data.get('final_price'):
            result += f"\nFinal Price: ${order_data['final_price']:,.2f}"
        elif order_data.get('agreed_price'):
            result += f"\nAgreed Price: ${order_data['agreed_price']:,.2f}"
        return result

    def _extract_preferences(self, message: str) -> Dict[str, Any]:
        """Extract user preferences from message"""
        preferences = {}
        message_lower = message.lower()
        
        # Extract budget preference
        if 'budget' in message_lower:
            price_pattern = r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
            prices = re.findall(price_pattern, message)
            if prices:
                preferences['max_budget'] = float(prices[0].replace(',', ''))
                
        # Extract fuel type preference
        if 'electric' in message_lower or 'ev' in message_lower:
            preferences['preferred_fuel_type'] = 'Electric'
        elif 'hybrid' in message_lower:
            preferences['preferred_fuel_type'] = 'Hybrid'
        elif 'gas' in message_lower:
            preferences['preferred_fuel_type'] = 'Gasoline'
            
        # Extract brand preference
        brands = ['toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi']
        for brand in brands:
            if brand in message_lower:
                preferences['preferred_brand'] = brand.title()
                break
                
        return preferences

    def _update_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Update user preferences in database"""
        # This is a simplified implementation
        # In a real system, you'd update a user preferences table
        logger.info(f"Updated user preferences: {preferences}")
        return len(preferences) > 0

    def _get_market_analysis(self) -> Optional[Dict[str, Any]]:
        """Get market analysis data"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Get average prices by brand
            cursor.execute("""
                SELECT brand, 
                       AVG(msrp) as avg_price,
                       COUNT(*) as model_count,
                       MIN(msrp) as min_price,
                       MAX(msrp) as max_price
                FROM vehicles 
                GROUP BY brand
                ORDER BY avg_price DESC
                LIMIT 10
            """)
            
            columns = [description[0] for description in cursor.description]
            brand_analysis = []
            
            for row in cursor.fetchall():
                brand_data = dict(zip(columns, row))
                brand_analysis.append(brand_data)
                
            conn.close()
            return {'brand_analysis': brand_analysis}
            
        except Exception as e:
            logger.error(f"Error getting market analysis: {e}")
            return None

    def _format_market_analysis(self, market_data: Dict[str, Any]) -> str:
        """Format market analysis for display"""
        if not market_data or 'brand_analysis' not in market_data:
            return "No market analysis data available."
            
        result = "Brand Price Analysis:\n"
        for brand_info in market_data['brand_analysis']:
            result += f"• {brand_info['brand']}: "
            result += f"Avg ${brand_info['avg_price']:,.0f} "
            result += f"(Range: ${brand_info['min_price']:,.0f} - ${brand_info['max_price']:,.0f})\n"
            result += f"  {brand_info['model_count']} models available\n"
            
        return result
    
    def _extract_query_parameters(self, state: MessageState) -> Dict[str, Any]:
        """Extract query parameters from orchestrator message or slots"""
        try:
            # Try to get from orchestrator message attributes first
            if hasattr(state, 'orchestrator_message') and state.orchestrator_message:
                attributes = getattr(state.orchestrator_message, 'attribute', {})
                if 'query_params' in attributes:
                    return json.loads(attributes['query_params']) if isinstance(attributes['query_params'], str) else attributes['query_params']
            
            # Try to get from slots (for coordination with CarAdvisorWorker)
            if hasattr(state, 'slots') and state.slots:
                if "car_advisor" in state.slots:
                    for slot in state.slots["car_advisor"]:
                        if slot.name == "db_query_params" and slot.value:
                            return json.loads(slot.value)
            
            # Extract from user message if available
            user_message = ""
            if state.user_message:
                user_message = state.user_message.message if hasattr(state.user_message, 'message') else str(state.user_message)
            
            # Parse basic criteria from user message
            search_criteria = self._extract_search_criteria(user_message)
            
            return {
                "query_types": ["search_vehicles"],
                "search_criteria": search_criteria,
                "result_limits": {
                    "vehicles": 8,
                    "dealers": 5,
                    "inventory": 10
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting query parameters: {e}")
            return {
                "query_types": ["search_vehicles"],
                "search_criteria": {},
                "result_limits": {"vehicles": 8, "dealers": 5, "inventory": 10}
            }
    
    def _format_coordinated_results(self, results: Dict[str, Any], query_params: Dict[str, Any]) -> str:
        """Format coordinated query results for comprehensive response"""
        response_parts = []
        
        # Header
        total_results = results.get("total_results", 0)
        query_types = query_params.get("query_types", [])
        response_parts.append(f"🔍 **Database Query Results** (Found {total_results} total items)")
        response_parts.append("")
        
        # Vehicle results
        vehicles = results.get("vehicles", [])
        if vehicles and "search_vehicles" in query_types:
            response_parts.append(f"**🚗 Vehicles ({len(vehicles)} found):**")
            for i, vehicle in enumerate(vehicles[:5], 1):  # Show top 5
                response_parts.append(f"{i}. {vehicle.get('year', 'N/A')} {vehicle.get('brand', 'N/A')} {vehicle.get('model', 'N/A')}")
                response_parts.append(f"   Price: ${vehicle.get('msrp', 0):,.0f} | Type: {vehicle.get('body_type', 'N/A')} | Fuel: {vehicle.get('fuel_type', 'N/A')}")
                mpg_city = vehicle.get('mpg_city', 0)
                mpg_highway = vehicle.get('mpg_highway', 0)
                if mpg_city and mpg_highway:
                    response_parts.append(f"   MPG: {mpg_city}/{mpg_highway} (city/highway)")
                response_parts.append("")
            
            if len(vehicles) > 5:
                response_parts.append(f"   ... and {len(vehicles) - 5} more vehicles")
                response_parts.append("")
        
        # Dealer results
        dealers = results.get("dealers", [])
        if dealers and "find_dealers" in query_types:
            response_parts.append(f"**🏢 Dealers ({len(dealers)} found):**")
            for dealer in dealers[:3]:  # Show top 3
                response_parts.append(f"• {dealer.get('name', 'N/A')}")
                response_parts.append(f"  Location: {dealer.get('city', 'N/A')}, {dealer.get('state', 'N/A')}")
                if dealer.get('phone'):
                    response_parts.append(f"  Phone: {dealer['phone']}")
                if dealer.get('customer_rating'):
                    response_parts.append(f"  Rating: {dealer['customer_rating']}/5.0")
                response_parts.append("")
        
        # Inventory results
        inventory = results.get("inventory", [])
        if inventory and "browse_inventory" in query_types:
            response_parts.append(f"**📦 Current Inventory ({len(inventory)} items):**")
            for item in inventory[:3]:  # Show top 3
                response_parts.append(f"• {item.get('year', 'N/A')} {item.get('brand', 'N/A')} {item.get('model', 'N/A')}")
                response_parts.append(f"  Status: {item.get('status', 'Available')} | Condition: {item.get('condition_type', 'N/A')}")
                response_parts.append(f"  Price: ${item.get('asking_price', item.get('msrp', 0)):,.0f}")
                response_parts.append("")
        
        # Market analysis
        market_analysis = results.get("market_analysis", {})
        if market_analysis and "analyze_market" in query_types:
            response_parts.append("**📊 Market Analysis:**")
            brand_analysis = market_analysis.get("brand_analysis", [])
            if brand_analysis:
                response_parts.append("Top brands by average price:")
                for brand_info in brand_analysis[:3]:
                    response_parts.append(f"• {brand_info['brand']}: Avg ${brand_info['avg_price']:,.0f} ({brand_info['model_count']} models)")
                response_parts.append("")
        
        # Summary
        if total_results == 0:
            response_parts.append("*No results found matching your criteria. Try adjusting your search parameters.*")
        else:
            response_parts.append(f"*Database query completed successfully. Found {total_results} items across {len(query_types)} search categories.*")
        
        return "\n".join(response_parts)
    
    # Public interface methods for CarAdvisorWorker coordination
    
    @classmethod
    def execute_coordinated_query(cls, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Public interface for CarAdvisorWorker to execute coordinated database queries"""
        try:
            worker = cls()
            worker._ensure_database_exists()
            
            results = {
                "vehicles": [],
                "dealers": [],
                "inventory": [],
                "market_analysis": {},
                "query_success": True,
                "total_results": 0
            }
            
            query_types = query_params.get("query_types", ["search_vehicles"])
            search_criteria = query_params.get("search_criteria", {})
            result_limits = query_params.get("result_limits", {})
            
            # Execute vehicle search if requested
            if "search_vehicles" in query_types:
                vehicles = worker._query_vehicles(search_criteria)
                results["vehicles"] = vehicles[:result_limits.get("vehicles", 10)]
                results["total_results"] += len(vehicles)
            
            # Execute dealer search if requested
            if "find_dealers" in query_types:
                location = search_criteria.get("location", "")
                dealers = worker._query_dealers(location)
                results["dealers"] = dealers[:result_limits.get("dealers", 5)]
            
            # Execute inventory search if requested
            if "browse_inventory" in query_types:
                inventory = worker._get_inventory_data()
                results["inventory"] = inventory[:result_limits.get("inventory", 10)]
            
            # Execute market analysis if requested
            if "analyze_market" in query_types:
                market_data = worker._get_market_analysis()
                if market_data:
                    results["market_analysis"] = market_data
            
            return results
            
        except Exception as e:
            logger.error(f"Error in coordinated query execution: {e}")
            return {
                "vehicles": [],
                "dealers": [],
                "inventory": [],
                "market_analysis": {},
                "query_success": False,
                "error": str(e),
                "total_results": 0
            } 