"""Tools for the customer service supervisor agent system.

This module provides tools for the customer service system, including mock implementations
for category and flow retrieval from an external API.
"""

from typing import Any, Dict, List

from langchain_core.tools import tool


@tool("get_request_categories")
def get_request_categories() -> List[Dict[str, Any]]:
    """Get the list of available request categories.

    This tool retrieves the list of categories that customer service requests can be classified into.
    Each category has an ID, name, and description.

    Returns:
        A list of dictionaries, each containing category information.
    """
    # Mock implementation - in a real system, this would call an external API
    categories = [
        {
            "id": "billing",
            "name": "Billing Issues",
            "description": "Issues related to billing, payments, refunds, and subscriptions.",
        },
        {
            "id": "technical",
            "name": "Technical Support",
            "description": "Technical issues with the product or service.",
        },
        {
            "id": "account",
            "name": "Account Management",
            "description": "Issues related to account creation, login, profile management, etc.",
        },
        {
            "id": "product",
            "name": "Product Information",
            "description": "Questions about product features, availability, or compatibility.",
        },
        {
            "id": "shipping",
            "name": "Shipping and Delivery",
            "description": "Issues related to shipping, delivery, tracking, or returns.",
        },
    ]
    return categories


@tool("get_category_flows")
def get_category_flows(category_id: str) -> List[Dict[str, Any]]:
    """Get the list of flows available for a specific category.

    This tool retrieves the list of flows that can be used to handle customer service requests
    in a specific category.

    Args:
        category_id: The ID of the category to get flows for.

    Returns:
        A list of dictionaries, each containing flow information.
    """
    # Mock implementation - in a real system, this would call an external API
    flows_by_category = {
        "billing": [
            {
                "id": "billing_refund",
                "name": "Process Refund Request",
                "description": "Handle customer requests for refunds.",
                "steps": [
                    "Verify customer identity and purchase details.",
                    "Check refund eligibility based on purchase date and company policy.",
                    "If eligible, process the refund and provide confirmation details.",
                    "If not eligible, explain the reason and offer alternative solutions.",
                ],
            },
            {
                "id": "billing_subscription",
                "name": "Subscription Management",
                "description": "Handle subscription changes, cancellations, or issues.",
                "steps": [
                    "Verify customer identity and subscription details.",
                    "Understand the specific subscription change or issue.",
                    "Process the requested change or troubleshoot the issue.",
                    "Confirm the changes and provide updated subscription details.",
                ],
            },
            {
                "id": "billing_payment",
                "name": "Payment Issue Resolution",
                "description": "Resolve issues with payments, declined cards, or billing errors.",
                "steps": [
                    "Identify the specific payment issue.",
                    "Verify customer identity and payment details.",
                    "Troubleshoot the payment issue and suggest solutions.",
                    "Process any necessary adjustments and confirm resolution.",
                ],
            },
        ],
        "technical": [
            {
                "id": "tech_troubleshooting",
                "name": "General Troubleshooting",
                "description": "General technical troubleshooting for common issues.",
                "steps": [
                    "Identify the specific technical issue.",
                    "Gather relevant system information and error details.",
                    "Guide the customer through basic troubleshooting steps.",
                    "If basic steps don't resolve the issue, escalate to specialized support.",
                ],
            },
            {
                "id": "tech_installation",
                "name": "Installation Support",
                "description": "Help with product installation or setup.",
                "steps": [
                    "Verify product version and system compatibility.",
                    "Guide the customer through the installation process step by step.",
                    "Troubleshoot any installation errors or issues.",
                    "Confirm successful installation and basic functionality.",
                ],
            },
        ],
        "account": [
            {
                "id": "account_reset",
                "name": "Password Reset",
                "description": "Help customers reset their password or recover account access.",
                "steps": [
                    "Verify customer identity using security questions or alternative contact methods.",
                    "Initiate the password reset process.",
                    "Guide the customer through creating a new secure password.",
                    "Confirm account access has been restored.",
                ],
            },
            {
                "id": "account_update",
                "name": "Account Information Update",
                "description": "Help customers update their account information.",
                "steps": [
                    "Verify customer identity.",
                    "Identify the specific information that needs to be updated.",
                    "Process the information update in the system.",
                    "Confirm the updates and verify the changes are reflected correctly.",
                ],
            },
        ],
        "product": [
            {
                "id": "product_info",
                "name": "Product Information",
                "description": "Provide detailed information about products or services.",
                "steps": [
                    "Identify the specific product or feature the customer is inquiring about.",
                    "Retrieve accurate and up-to-date information about the product.",
                    "Present the information in a clear and helpful manner.",
                    "Address any follow-up questions or comparisons with other products.",
                ],
            },
            {
                "id": "product_compatibility",
                "name": "Compatibility Check",
                "description": "Check if a product is compatible with customer's system or other products.",
                "steps": [
                    "Gather information about the customer's system or existing products.",
                    "Check compatibility requirements and limitations.",
                    "Provide a clear compatibility assessment.",
                    "If incompatible, suggest alternatives or workarounds.",
                ],
            },
        ],
        "shipping": [
            {
                "id": "shipping_status",
                "name": "Order Tracking",
                "description": "Help customers track their orders and shipments.",
                "steps": [
                    "Verify customer identity and order details.",
                    "Retrieve the current shipping status and location.",
                    "Provide estimated delivery date and tracking information.",
                    "Address any concerns about delays or shipping issues.",
                ],
            },
            {
                "id": "shipping_return",
                "name": "Return Processing",
                "description": "Guide customers through the return process.",
                "steps": [
                    "Verify order details and return eligibility.",
                    "Explain the return process and policy.",
                    "Generate return labels or instructions.",
                    "Process the return request and provide confirmation details.",
                ],
            },
        ],
    }

    return flows_by_category.get(category_id, [])


@tool("get_flow_details")
def get_flow_details(flow_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific flow.

    This tool retrieves detailed information about a specific flow, including its steps,
    required information, and any special handling instructions.

    Args:
        flow_id: The ID of the flow to get details for.

    Returns:
        A dictionary containing detailed flow information.
    """
    # Mock implementation - in a real system, this would call an external API
    all_flows = []
    for category_flows in (
        get_category_flows.__annotations__["return"]
        .__args__[0]
        .__args__[0]
        .__args__[0]
        .values()
    ):  # type: ignore
        all_flows.extend(category_flows)

    for flow in all_flows:
        if flow["id"] == flow_id:
            # Add more detailed information for the flow
            flow["detailed_instructions"] = (
                f"Detailed instructions for handling {flow['name']} requests."
            )
            flow["required_information"] = [
                "Customer ID",
                "Order details",
                "Specific issue description",
            ]
            flow["expected_outcome"] = (
                f"Successfully resolve the {flow['name'].lower()} request to customer satisfaction."
            )
            return flow

    return {"error": f"Flow with ID '{flow_id}' not found."}


@tool("set_ticket_info")
def set_ticket_info(ticket_id: str, info_dict: Dict[str, Any]) -> str:
    """Set detailed information about a specific flow.

    This tool sets detailed information about a specific flow, including its steps,
    required information, and any special handling instructions.

    Args:
        ticket_id: The ID of the ticket to set details for.
        info_dict: A dictionary containing the information to set.

    Returns:
        A string indicating that the ticket info has been set.
    """

    return "Ticket info set"

@tool("get_knowledge_info")
def get_knowledge_info(query: str) -> str:
    """Get information from the knowledge base.

    This tool retrieves information from the knowledge base based on the provided query.

    Args:
        query: The query to search the knowledge base for.

    Returns:
        A string containing the information from the knowledge base.
    """

    return "Knowledge info retrieved"


# List of all tools available for the supervisor agent
SUPERVISOR_TOOLS = [
    get_request_categories,
    get_category_flows,
    get_flow_details,
]
