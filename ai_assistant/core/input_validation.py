"""
Comprehensive Input Validation Framework for AI Assistant

Provides centralized validation for all input types across the system:
- Web API endpoints
- WebSocket messages  
- CLI command inputs
- Configuration parameters
- File uploads and data imports

Includes sanitization, type checking, and security validation.
"""

import re
import json
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass
from enum import Enum
import ipaddress
import email.utils
from pathlib import Path

try:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from core.audit_logger import audit_security_event, SeverityLevel
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


class ValidationError(Exception):
    """Custom exception for validation failures"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)


class InputType(Enum):
    """Types of input validation"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    IP_ADDRESS = "ip_address"
    FILE_PATH = "file_path"
    JSON = "json"
    HTML = "html"
    COMMAND = "command"
    API_KEY = "api_key"
    PIN = "pin"
    PHONE = "phone"
    DATE = "date"
    UUID = "uuid"


@dataclass
class ValidationRule:
    """Validation rule configuration"""
    field_name: str
    input_type: InputType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    sanitize: bool = True
    description: str = ""


class InputValidator:
    """
    Comprehensive input validation system
    
    Features:
    - Type-specific validation
    - Security sanitization
    - Pattern matching
    - Range validation
    - Custom validation functions
    - XSS and injection prevention
    """
    
    def __init__(self):
        """Initialize validator with security patterns"""
        # Common security patterns
        self.sql_injection_patterns = [
            r"('|(\\')|(;|\\x27|\\x3D))",
            r"union\s+select",
            r"select\s+.*\s+from",
            r"drop\s+table",
            r"delete\s+from",
            r"insert\s+into",
            r"update\s+.*\s+set"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe",
            r"<object",
            r"<embed"
        ]
        
        self.command_injection_patterns = [
            r"[;&|`$()]",
            r"../",
            r"\.\./",
            r"\\\\",
            r"%2e%2e",
            r"cmd\.exe",
            r"/bin/",
            r"powershell"
        ]
        
        # File extension restrictions
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
            '.jar', '.ps1', '.sh', '.php', '.asp', '.jsp', '.py'
        }
        
        # Common validation patterns
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            'pin': r'^\d{4,8}$',
            'phone': r'^\+?[\d\s\-\(\)]{7,15}$',
            'api_key': r'^[A-Za-z0-9_\-]{16,128}$',
            'safe_filename': r'^[a-zA-Z0-9._\-\s]{1,255}$',
            'command_safe': r'^[a-zA-Z0-9\s\-_.:/\\]+$'
        }
    
    def validate_field(self, value: Any, rule: ValidationRule) -> Any:
        """
        Validate a single field against its rule
        
        Args:
            value: Value to validate
            rule: Validation rule
            
        Returns:
            Validated and sanitized value
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check if field is required
            if value is None or value == "":
                if rule.required:
                    raise ValidationError(f"Field '{rule.field_name}' is required", rule.field_name, value)
                return None
            
            # Type-specific validation
            validated_value = self._validate_type(value, rule)
            
            # Length validation for strings
            if rule.input_type == InputType.STRING and isinstance(validated_value, str):
                if rule.min_length and len(validated_value) < rule.min_length:
                    raise ValidationError(
                        f"Field '{rule.field_name}' must be at least {rule.min_length} characters",
                        rule.field_name, value
                    )
                if rule.max_length and len(validated_value) > rule.max_length:
                    raise ValidationError(
                        f"Field '{rule.field_name}' must not exceed {rule.max_length} characters",
                        rule.field_name, value
                    )
            
            # Range validation for numbers
            if rule.input_type in [InputType.INTEGER, InputType.FLOAT]:
                if rule.min_value is not None and validated_value < rule.min_value:
                    raise ValidationError(
                        f"Field '{rule.field_name}' must be at least {rule.min_value}",
                        rule.field_name, value
                    )
                if rule.max_value is not None and validated_value > rule.max_value:
                    raise ValidationError(
                        f"Field '{rule.field_name}' must not exceed {rule.max_value}",
                        rule.field_name, value
                    )
            
            # Pattern validation
            if rule.pattern and isinstance(validated_value, str):
                if not re.match(rule.pattern, validated_value, re.IGNORECASE):
                    raise ValidationError(
                        f"Field '{rule.field_name}' format is invalid",
                        rule.field_name, value
                    )
            
            # Allowed values validation
            if rule.allowed_values and validated_value not in rule.allowed_values:
                raise ValidationError(
                    f"Field '{rule.field_name}' must be one of: {rule.allowed_values}",
                    rule.field_name, value
                )
            
            # Custom validation
            if rule.custom_validator and not rule.custom_validator(validated_value):
                raise ValidationError(
                    f"Field '{rule.field_name}' failed custom validation",
                    rule.field_name, value
                )
            
            # Security checks
            if isinstance(validated_value, str):
                self._check_security_threats(validated_value, rule.field_name)
            
            # Sanitization
            if rule.sanitize and isinstance(validated_value, str):
                validated_value = self._sanitize_string(validated_value, rule.input_type)
            
            return validated_value
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Validation error for field {rule.field_name}: {e}")
            raise ValidationError(f"Validation failed for field '{rule.field_name}': {str(e)}", rule.field_name, value)
    
    def _validate_type(self, value: Any, rule: ValidationRule) -> Any:
        """Validate and convert value to expected type"""
        try:
            if rule.input_type == InputType.STRING:
                return str(value)
            
            elif rule.input_type == InputType.INTEGER:
                if isinstance(value, str):
                    return int(value)
                elif isinstance(value, (int, float)):
                    return int(value)
                else:
                    raise ValidationError(f"Cannot convert {type(value)} to integer")
            
            elif rule.input_type == InputType.FLOAT:
                if isinstance(value, str):
                    return float(value)
                elif isinstance(value, (int, float)):
                    return float(value)
                else:
                    raise ValidationError(f"Cannot convert {type(value)} to float")
            
            elif rule.input_type == InputType.BOOLEAN:
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(value, int):
                    return bool(value)
                else:
                    raise ValidationError(f"Cannot convert {type(value)} to boolean")
            
            elif rule.input_type == InputType.EMAIL:
                email_str = str(value).lower().strip()
                if not re.match(self.patterns['email'], email_str):
                    raise ValidationError("Invalid email format")
                return email_str
            
            elif rule.input_type == InputType.URL:
                url_str = str(value).strip()
                if not re.match(self.patterns['url'], url_str):
                    raise ValidationError("Invalid URL format")
                return url_str
            
            elif rule.input_type == InputType.IP_ADDRESS:
                try:
                    ip = ipaddress.ip_address(str(value).strip())
                    return str(ip)
                except ValueError:
                    raise ValidationError("Invalid IP address format")
            
            elif rule.input_type == InputType.FILE_PATH:
                path_str = str(value).strip()
                # Security check for path traversal
                if '..' in path_str or path_str.startswith('/') or '\\\\' in path_str:
                    raise ValidationError("Invalid file path: potential directory traversal")
                
                # Check file extension
                path = Path(path_str)
                if path.suffix.lower() in self.dangerous_extensions:
                    raise ValidationError(f"Dangerous file extension: {path.suffix}")
                
                return path_str
            
            elif rule.input_type == InputType.JSON:
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError as e:
                        raise ValidationError(f"Invalid JSON: {e}")
                else:
                    return value  # Already parsed
            
            elif rule.input_type == InputType.PIN:
                pin_str = str(value).strip()
                if not re.match(self.patterns['pin'], pin_str):
                    raise ValidationError("PIN must be 4-8 digits")
                return pin_str
            
            elif rule.input_type == InputType.API_KEY:
                key_str = str(value).strip()
                if not re.match(self.patterns['api_key'], key_str):
                    raise ValidationError("Invalid API key format")
                return key_str
            
            elif rule.input_type == InputType.UUID:
                uuid_str = str(value).lower().strip()
                if not re.match(self.patterns['uuid'], uuid_str):
                    raise ValidationError("Invalid UUID format")
                return uuid_str
            
            elif rule.input_type == InputType.COMMAND:
                cmd_str = str(value).strip()
                if not re.match(self.patterns['command_safe'], cmd_str):
                    raise ValidationError("Command contains unsafe characters")
                return cmd_str
            
            else:
                return str(value)
                
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Type conversion failed: {e}")
    
    def _check_security_threats(self, value: str, field_name: str):
        """Check for common security threats in string inputs"""
        value_lower = value.lower()
        
        # SQL Injection detection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value_lower, re.IGNORECASE):
                if AUDIT_AVAILABLE:
                    audit_security_event(
                        f"SQL injection attempt detected in field {field_name}: {value[:100]}",
                        SeverityLevel.HIGH
                    )\n                raise ValidationError(f"Potential SQL injection detected in field '{field_name}'")\n        \n        # XSS detection\n        for pattern in self.xss_patterns:\n            if re.search(pattern, value_lower, re.IGNORECASE):\n                if AUDIT_AVAILABLE:\n                    audit_security_event(\n                        f\"XSS attempt detected in field {field_name}: {value[:100]}\",\n                        SeverityLevel.HIGH\n                    )\n                raise ValidationError(f\"Potential XSS attack detected in field '{field_name}'\")\n        \n        # Command injection detection\n        for pattern in self.command_injection_patterns:\n            if re.search(pattern, value, re.IGNORECASE):\n                if AUDIT_AVAILABLE:\n                    audit_security_event(\n                        f\"Command injection attempt detected in field {field_name}: {value[:100]}\",\n                        SeverityLevel.HIGH\n                    )\n                raise ValidationError(f\"Potential command injection detected in field '{field_name}'\")\n    \n    def _sanitize_string(self, value: str, input_type: InputType) -> str:\n        \"\"\"Sanitize string input based on type\"\"\"\n        if input_type == InputType.HTML:\n            # HTML sanitization (escape dangerous characters)\n            return html.escape(value, quote=True)\n        \n        elif input_type == InputType.URL:\n            # URL encoding for unsafe characters\n            return urllib.parse.quote(value, safe=\":/?#[]@!$&'()*+,;=\")\n        \n        elif input_type == InputType.STRING:\n            # Basic string sanitization\n            # Remove null bytes and control characters\n            sanitized = re.sub(r'[\\x00-\\x1f\\x7f-\\x9f]', '', value)\n            # Normalize whitespace\n            sanitized = ' '.join(sanitized.split())\n            return sanitized.strip()\n        \n        else:\n            return value.strip()\n    \n    def validate_dict(self, data: Dict[str, Any], rules: List[ValidationRule]) -> Dict[str, Any]:\n        \"\"\"\n        Validate a dictionary against a set of rules\n        \n        Args:\n            data: Dictionary to validate\n            rules: List of validation rules\n            \n        Returns:\n            Validated and sanitized dictionary\n            \n        Raises:\n            ValidationError: If any field fails validation\n        \"\"\"\n        validated_data = {}\n        errors = []\n        \n        # Create rule lookup\n        rule_map = {rule.field_name: rule for rule in rules}\n        \n        # Validate each field\n        for field_name, rule in rule_map.items():\n            try:\n                value = data.get(field_name)\n                validated_value = self.validate_field(value, rule)\n                if validated_value is not None:  # Don't include None values\n                    validated_data[field_name] = validated_value\n            except ValidationError as e:\n                errors.append(e)\n        \n        # Check for unexpected fields\n        expected_fields = set(rule_map.keys())\n        provided_fields = set(data.keys())\n        unexpected_fields = provided_fields - expected_fields\n        \n        if unexpected_fields:\n            logger.warning(f\"Unexpected fields in input: {unexpected_fields}\")\n            # Optionally reject unexpected fields\n            # errors.append(ValidationError(f\"Unexpected fields: {unexpected_fields}\"))\n        \n        if errors:\n            # Combine error messages\n            error_messages = [str(e) for e in errors]\n            raise ValidationError(f\"Validation failed: {'; '.join(error_messages)}\")\n        \n        return validated_data\n    \n    def validate_api_request(self, data: Dict[str, Any], endpoint: str) -> Dict[str, Any]:\n        \"\"\"Validate API request data based on endpoint\"\"\"\n        rules = self._get_api_rules(endpoint)\n        return self.validate_dict(data, rules)\n    \n    def _get_api_rules(self, endpoint: str) -> List[ValidationRule]:\n        \"\"\"Get validation rules for specific API endpoints\"\"\"\n        # Common API validation rules\n        common_rules = {\n            '/api/auth/login': [\n                ValidationRule('pin', InputType.PIN, required=True, description='User PIN')\n            ],\n            '/api/chat': [\n                ValidationRule('message', InputType.STRING, required=True, max_length=10000, description='Chat message'),\n                ValidationRule('session_id', InputType.UUID, required=False, description='Session identifier'),\n                ValidationRule('model_preference', InputType.STRING, required=False, \n                             allowed_values=['gpt-4', 'gpt-3.5-turbo', 'claude', 'gemini'], description='AI model preference')\n            ],\n            '/api/system/command': [\n                ValidationRule('command', InputType.COMMAND, required=True, max_length=500, description='System command'),\n                ValidationRule('user_id', InputType.STRING, required=True, description='User identifier')\n            ],\n            '/api/settings': [\n                ValidationRule('setting_name', InputType.STRING, required=True, max_length=100, description='Setting name'),\n                ValidationRule('setting_value', InputType.STRING, required=True, max_length=1000, description='Setting value')\n            ],\n            '/api/file/upload': [\n                ValidationRule('filename', InputType.FILE_PATH, required=True, description='File name'),\n                ValidationRule('content_type', InputType.STRING, required=True, \n                             allowed_values=['text/plain', 'application/json', 'image/jpeg', 'image/png'],\n                             description='File content type')\n            ]\n        }\n        \n        return common_rules.get(endpoint, [])\n\n\nclass WebSocketValidator:\n    \"\"\"Specialized validator for WebSocket messages\"\"\"\n    \n    def __init__(self, input_validator: InputValidator):\n        self.validator = input_validator\n        \n        # WebSocket message types and their validation rules\n        self.message_rules = {\n            'chat': [\n                ValidationRule('type', InputType.STRING, required=True, allowed_values=['chat']),\n                ValidationRule('message', InputType.STRING, required=True, max_length=10000),\n                ValidationRule('session_id', InputType.STRING, required=False)\n            ],\n            'command': [\n                ValidationRule('type', InputType.STRING, required=True, allowed_values=['command']),\n                ValidationRule('command', InputType.STRING, required=True, max_length=500),\n                ValidationRule('parameters', InputType.JSON, required=False)\n            ],\n            'system': [\n                ValidationRule('type', InputType.STRING, required=True, allowed_values=['system']),\n                ValidationRule('action', InputType.STRING, required=True, \n                             allowed_values=['status', 'stats', 'config']),\n                ValidationRule('data', InputType.JSON, required=False)\n            ]\n        }\n    \n    def validate_message(self, message: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Validate WebSocket message\"\"\"\n        # Validate message structure\n        if not isinstance(message, dict):\n            raise ValidationError(\"WebSocket message must be a JSON object\")\n        \n        message_type = message.get('type')\n        if not message_type:\n            raise ValidationError(\"WebSocket message must include 'type' field\")\n        \n        rules = self.message_rules.get(message_type, [])\n        if not rules:\n            raise ValidationError(f\"Unknown WebSocket message type: {message_type}\")\n        \n        return self.validator.validate_dict(message, rules)\n\n\nclass CLIValidator:\n    \"\"\"Specialized validator for CLI command inputs\"\"\"\n    \n    def __init__(self, input_validator: InputValidator):\n        self.validator = input_validator\n    \n    def validate_command_args(self, args: List[str]) -> List[str]:\n        \"\"\"Validate CLI command arguments\"\"\"\n        validated_args = []\n        \n        for arg in args:\n            # Basic sanitization\n            if not isinstance(arg, str):\n                raise ValidationError(f\"Command argument must be string, got {type(arg)}\")\n            \n            # Check for injection attempts\n            self.validator._check_security_threats(arg, 'cli_argument')\n            \n            # Sanitize\n            sanitized_arg = self.validator._sanitize_string(arg, InputType.COMMAND)\n            validated_args.append(sanitized_arg)\n        \n        return validated_args\n    \n    def validate_file_path(self, file_path: str) -> str:\n        \"\"\"Validate file path for CLI operations\"\"\"\n        rule = ValidationRule('file_path', InputType.FILE_PATH, required=True)\n        return self.validator.validate_field(file_path, rule)\n\n\n# Global validator instances\n_input_validator = None\n_websocket_validator = None\n_cli_validator = None\n\ndef get_input_validator() -> InputValidator:\n    \"\"\"Get global input validator instance\"\"\"\n    global _input_validator\n    if _input_validator is None:\n        _input_validator = InputValidator()\n    return _input_validator\n\ndef get_websocket_validator() -> WebSocketValidator:\n    \"\"\"Get WebSocket validator instance\"\"\"\n    global _websocket_validator\n    if _websocket_validator is None:\n        _websocket_validator = WebSocketValidator(get_input_validator())\n    return _websocket_validator\n\ndef get_cli_validator() -> CLIValidator:\n    \"\"\"Get CLI validator instance\"\"\"\n    global _cli_validator\n    if _cli_validator is None:\n        _cli_validator = CLIValidator(get_input_validator())\n    return _cli_validator\n\n\n# Convenience functions for common validation scenarios\ndef validate_api_input(data: Dict[str, Any], endpoint: str) -> Dict[str, Any]:\n    \"\"\"Validate API input data\"\"\"\n    return get_input_validator().validate_api_request(data, endpoint)\n\ndef validate_websocket_message(message: Dict[str, Any]) -> Dict[str, Any]:\n    \"\"\"Validate WebSocket message\"\"\"\n    return get_websocket_validator().validate_message(message)\n\ndef validate_cli_command(args: List[str]) -> List[str]:\n    \"\"\"Validate CLI command arguments\"\"\"\n    return get_cli_validator().validate_command_args(args)\n\ndef validate_pin(pin: str) -> str:\n    \"\"\"Validate PIN format\"\"\"\n    rule = ValidationRule('pin', InputType.PIN, required=True)\n    return get_input_validator().validate_field(pin, rule)\n\ndef validate_email(email: str) -> str:\n    \"\"\"Validate email address\"\"\"\n    rule = ValidationRule('email', InputType.EMAIL, required=True)\n    return get_input_validator().validate_field(email, rule)\n\ndef validate_file_upload(filename: str, content_type: str) -> Dict[str, str]:\n    \"\"\"Validate file upload parameters\"\"\"\n    rules = [\n        ValidationRule('filename', InputType.FILE_PATH, required=True),\n        ValidationRule('content_type', InputType.STRING, required=True)\n    ]\n    data = {'filename': filename, 'content_type': content_type}\n    return get_input_validator().validate_dict(data, rules)\n\n\nif __name__ == \"__main__\":\n    # Test input validation system\n    print(\"Testing input validation system...\")\n    \n    validator = InputValidator()\n    \n    # Test API validation\n    try:\n        login_data = {'pin': '1234'}\n        validated = validate_api_input(login_data, '/api/auth/login')\n        print(f\"Valid login data: {validated}\")\n    except ValidationError as e:\n        print(f\"Login validation error: {e}\")\n    \n    # Test WebSocket validation\n    try:\n        ws_message = {\n            'type': 'chat',\n            'message': 'Hello, assistant!',\n            'session_id': 'abc123'\n        }\n        validated_ws = validate_websocket_message(ws_message)\n        print(f\"Valid WebSocket message: {validated_ws}\")\n    except ValidationError as e:\n        print(f\"WebSocket validation error: {e}\")\n    \n    # Test security detection\n    try:\n        malicious_input = \"'; DROP TABLE users; --\"\n        validator._check_security_threats(malicious_input, 'test_field')\n    except ValidationError as e:\n        print(f\"Security threat detected: {e}\")\n    \n    # Test file validation\n    try:\n        safe_file = validate_file_upload('document.pdf', 'application/pdf')\n        print(f\"Valid file upload: {safe_file}\")\n    except ValidationError as e:\n        print(f\"File validation error: {e}\")\n    \n    print(\"✅ Input validation test completed!\")"
    