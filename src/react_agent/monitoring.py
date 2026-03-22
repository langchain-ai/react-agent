import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
import logging

logger = logging.getLogger(__name__)

def init_monitoring(project_name: str = "airflow-sre-agent"):
    """
    Initializes Arize Phoenix for LangChain/LangGraph monitoring.
    
    This function:
    1. Registers an OpenTelemetry tracer for Phoenix.
    2. Instruments LangChain and LangGraph.
    """
    try:
        # 1. Register the OpenTelemetry tracer
        # This will send traces to a running Phoenix instance (default localhost:6006)
        print(f"\n🔭 Arize Phoenix Tracing Initialized (Project: {project_name})")
        register(project_name=project_name)
        
        # 2. Instrument LangChain
        LangChainInstrumentor().instrument()
    except Exception as e:
        logger.error(f"Failed to initialize Phoenix monitoring: {e}")
        return None
