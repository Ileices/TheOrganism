BUILD_STEP_SCHEMA = {
    "type": "object",
    "required": ["step_number", "tasks"],
    "properties": {
        "step_number": {"type": "integer"},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action", "path"],
                "properties": {
                    "action": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                }
            }
        }
    }
}

def validate_schema(data: dict, schema_type: str) -> bool:
    """Validate data against specified schema"""
    # ...existing code...
