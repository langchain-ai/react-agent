import copy
from typing import Dict, List, Optional, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from tw_ai_agents.tools.base_agent_tools import BaseAgentWithTools

llm = ChatOpenAI(model="gpt-4o")


class DPDAgentTools(BaseAgentWithTools):
    """
    Agent that provides tools for tracking deliveries and packages from DPD.
    
    This agent can be used to retrieve delivery status and nearest dropoff locations.
    """
    
    def __init__(self) -> None:
        node_name = "dpd_delivery_tracker"
        base_system_prompt = """You are a helpful assistant that can track deliveries and packages.
        Your goal is to provide accurate information about delivery status, estimated delivery dates, 
        and other details by using the available tools to retrieve package tracking information.
        """
        description = "You are a helpful assistant that can track deliveries and provide package information."

        super().__init__(
            system_prompt=base_system_prompt,
            node_name=node_name,
            description=description,
        )

    @tool
    def track_package(self, tracking_number: str) -> Dict[str, Any]:
        """
        Track a package using its tracking number.
        
        This tool retrieves the current status, estimated delivery date, location,
        and tracking history for a package from the specified delivery company.
        
        Args:
            tracking_number: The tracking number of the package
        
        Returns:
            A dictionary containing delivery information including:
            - current_status: Current status of the package
            - estimated_delivery: Estimated delivery date and time
            - location: Current location of the package
            - tracking_history: List of tracking events with timestamps
        """
        # Mock response - in a real implementation, this would call an API
        base_mock_delivery_info = {
            "tracking_number": tracking_number,
            "current_status": "In Transit",
            "estimated_delivery": "2023-11-15T18:00:00",
            "location": "Sorting Center Zurich",
            "tracking_history": [ 
                {"timestamp": "2023-11-12T09:15:00", "status": "Package received", "location": "Distribution Center Geneva"},
                {"timestamp": "2023-11-13T14:30:00", "status": "In Transit", "location": "En route to Zurich"},
                {"timestamp": "2023-11-14T08:45:00", "status": "In Transit", "location": "Sorting Center Zurich"}
            ],
        }
        
        mock_delivery_info = {
            "123456": {
                "current_status": "Lost package",
                "estimated_delivery": None,
                "location": "Unknown",
                "tracking_history": [
                    {"timestamp": "2023-11-12T09:15:00", "status": "Package received", "location": "Distribution Center Geneva"},
                ],
            }
        }
        
        return mock_delivery_info.get(tracking_number, base_mock_delivery_info)
    
    @tool
    def find_nearest_dropoff(self, postal_code: str) -> Dict[str, Any]:
        """
        Find the nearest package dropoff locations based on postal code.
        
        Args:
            postal_code: The postal code to search near
            
        Returns:
            A dictionary containing dropoff locations including:
            - locations: List of nearby dropoff points with address and hours
            - postal_code: The searched postal code
        """
        # Mock response with nearby locations
        mock_locations = [
            {
                "name": "Central Post Office",
                "address": "Bahnhofstrasse 15, 8001 Zurich",
                "distance": "0.5 km",
                "hours": "Mon-Fri: 8:00-18:00, Sat: 9:00-16:00",
                "services": ["Package Dropoff", "Package Pickup", "Mail Services"]
            },
            {
                "name": "Shopping Center Dropoff Point",
                "address": "Shopville, Bahnhofplatz, 8001 Zurich",
                "distance": "0.7 km",
                "hours": "Mon-Sat: 9:00-20:00, Sun: Closed",
                "services": ["Package Dropoff", "Package Pickup"]
            },
            {
                "name": "Neighborhood Service Center",
                "address": "Langstrasse 45, 8004 Zurich",
                "distance": "1.2 km",
                "hours": "Mon-Fri: 9:00-19:00, Sat: 10:00-16:00",
                "services": ["Package Dropoff", "Mail Services"]
            }
        ]
        
        return {"locations": mock_locations}
    
    def get_tools(self) -> List:
        """Return all available tools for the delivery agent."""
        return [
            self.track_package,
            self.find_nearest_dropoff,
        ] 