import asyncio
import os

from dotenv import load_dotenv

from react_agent.context import Context
from react_agent.graph import graph
from react_agent.monitoring import init_monitoring

load_dotenv()

# os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

async def main():
    # Initialize Airze Phoenix Monitoring (Open-source alternative to LangSmith)
    init_monitoring()

    sample_log_1 = """
    Caused by: com.google.cloud.spark.bigquery.repackaged.com.google.cloud.bigquery.BigQueryException: Error while reading data, error message: Schema mismatch: referenced variable 'df.list.element.PRODUCT_ID' has array levels of 1, while the corresponding field path to Parquet column has 0 repeated fields File: gs://my-recommendation/temp/.spark-bigquery-application_1683269466816_0014-881c60d3-da1a-44a3-92e3-a10de968b71c/part-00102-7f5d2b2b-22e2-4cf8-98fc-866eacc9c9db-c000.snappy.parquet
	at com.google.cloud.spark.bigquery.repackaged.com.google.cloud.bigquery.Job.reload(Job.java:419)
    """

    sample_log_2 = """
        [2024-08-08, 14:41:09 EDT] {file_task_handler.py:522} ERROR - Could not read served logs
        Traceback (most recent call last):
        File "/home/airflow/.local/lib/python3.8/site-packages/airflow/models/taskinstance.py", line 1407, in _run_raw_task
            self._execute_task_with_callbacks(context, test_mode)
        File "/home/airflow/.local/lib/python3.8/site-packages/airflow/models/taskinstance.py", line 1558, in _execute_task_with_callbacks
            result = self._execute_task(context, task_orig)
        File "/home/airflow/.local/lib/python3.8/site-packages/airflow/models/taskinstance.py", line 1628, in _execute_task
            result = execute_callable(context=context)
        File "/home/airflow/.local/lib/python3.8/site-packages/airflow/operators/bash.py", line 210, in execute
            raise AirflowException(
        airflow.exceptions.AirflowException: Bash command failed. The command returned a non-zero exit code 1.

    """

    sample_log_3 = """
        [2024-08-08, 14:41:09 EDT] {file_task_handler.py:522} ERROR - Could not read served logs
        Traceback (most recent call last):
        File "/home/airflow/.local/lib/python3.8/site-packages/airflow/models/taskinstance.py", line 1407, in _run_raw_task
            self._execute_task_with_callbacks(context, test_mode)
        File "/home/airflow/.local/lib/python3.8/site-packages/airflow/models/taskinstance.py", line 1558, in _execute_task_with_callbacks
            result = self._execute_task(context, task_orig)
        File "/home/airflow/.local/lib/python3.8/site-packages/airflow/models/taskinstance.py", line 1628, in _execute_task
            result = execute_callable(context=context)
        File "/home/airflow/.local/lib/python3.8/site-packages/airflow/operators/bash.py", line 210, in execute
            raise AirflowException("Kubernetes pod TEST_SQL_EXETUION was deleted before it finished")
        [2024-08-08, 14:41:09 EDT] {kubernetes_executor.py:112} ERROR - Error while fetching logs from Kubernetes pod:         
    """

    sample_log = sample_log_2

    inputs = {
        "messages": [
            (
                "user",
                f"Analyze this Airflow log and find the root cause. Please provide a detailed JSON report: {sample_log}",
            )
        ],
        "raw_log": sample_log,
    }

    # MODEL = "google_genai/gemini-3-flash-preview"
    MODEL = "ollama/qwen2.5-coder:7b"
    print(f"Sending Airflow Log to agent with {MODEL}...")
    try:
        async for chunk in graph.astream(
            inputs,
            stream_mode="values",            
            context=Context(model=MODEL),
        ):
            message = chunk["messages"][-1]
            message_type = getattr(message, "type", "unknown")
            if hasattr(message, "content") and getattr(message, "content", ""):
                print(f"[{message_type}]: {message.content}")
            if hasattr(message, "tool_calls") and getattr(message, "tool_calls", None):
                print(f"[{message_type}] (tool call): {message.tool_calls}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during execution: {e}")


if __name__ == "__main__":
    asyncio.run(main())
