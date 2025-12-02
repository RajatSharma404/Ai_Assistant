"""
Security Audit Logging System for AI Assistant

Provides comprehensive logging of security events, user actions, and system operations.
Includes log analysis, alerting, and compliance reporting capabilities.
"""

import json
import datetime
import os
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Enum, Union
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import sqlite3
from contextlib import contextmanager

try:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from core.encryption import get_encryption
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    logger.warning("Encryption not available for audit logs")


class EventType(Enum):
    """Types of audit events"""
    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_PIN_CHANGE = "auth.pin.change"
    AUTH_SESSION_EXPIRED = "auth.session.expired"
    
    # System operations
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_CONFIG_CHANGE = "system.config.change"
    SYSTEM_FILE_ACCESS = "system.file.access"
    SYSTEM_COMMAND_EXEC = "system.command.exec"
    SYSTEM_APP_LAUNCH = "system.app.launch"
    
    # API operations
    API_REQUEST = "api.request"
    API_ERROR = "api.error"
    API_RATE_LIMIT = "api.rate_limit"
    API_KEY_USAGE = "api.key.usage"
    
    # Data operations
    DATA_ACCESS = "data.access"
    DATA_MODIFICATION = "data.modification"
    DATA_DELETION = "data.deletion"
    DATA_EXPORT = "data.export"
    
    # Security events
    SECURITY_INTRUSION_ATTEMPT = "security.intrusion.attempt"
    SECURITY_PERMISSION_DENIED = "security.permission.denied"
    SECURITY_ENCRYPTION_ERROR = "security.encryption.error"
    SECURITY_SUSPICIOUS_ACTIVITY = "security.suspicious.activity"
    
    # User interactions
    USER_COMMAND = "user.command"
    USER_CHAT_MESSAGE = "user.chat.message"
    USER_SETTINGS_CHANGE = "user.settings.change"
    USER_PREFERENCE_UPDATE = "user.preference.update"


class SeverityLevel(Enum):
    """Event severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AuditEvent:
    """Audit event structure"""
    event_id: str
    timestamp: datetime.datetime
    event_type: EventType
    severity: SeverityLevel
    user_id: str
    session_id: Optional[str]
    source_ip: Optional[str]
    user_agent: Optional[str]
    message: str
    details: Dict[str, Any]
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['event_type'] = self.event_type.value
        result['severity'] = self.severity.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
        data['event_type'] = EventType(data['event_type'])
        data['severity'] = SeverityLevel(data['severity'])
        return cls(**data)


class AuditLogger:
    """
    Comprehensive audit logging system
    
    Features:
    - Structured event logging with encryption
    - Real-time threat detection
    - Compliance reporting
    - Log integrity verification
    - Automatic log rotation and archival
    """
    
    def __init__(self, log_dir: str = "logs/security", encrypt_logs: bool = True):
        """
        Initialize audit logger
        
        Args:
            log_dir: Directory for audit logs
            encrypt_logs: Whether to encrypt sensitive log data
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.encrypt_logs = encrypt_logs and ENCRYPTION_AVAILABLE
        
        # Database for structured logging
        self.db_path = self.log_dir / "audit.db"
        self._init_database()
        
        # Event queue for asynchronous logging
        self.event_queue = Queue(maxsize=10000)
        self.processing_thread = None
        self.running = True
        
        # Alert thresholds
        self.alert_thresholds = {
            EventType.AUTH_LOGIN_FAILURE: 3,  # 3 failed logins
            EventType.SECURITY_INTRUSION_ATTEMPT: 1,  # Any intrusion attempt
            EventType.API_ERROR: 10,  # 10 API errors
        }
        
        # Recent events cache for pattern detection
        self.recent_events = []
        self.cache_lock = threading.Lock()
        
        # Start background processing
        self.start_processing()
        
        logger.info(f"Audit logger initialized at {self.log_dir}")
    
    def _init_database(self):
        """Initialize audit database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Audit events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp DATETIME,
                        event_type TEXT,
                        severity TEXT,
                        user_id TEXT,
                        session_id TEXT,
                        source_ip TEXT,
                        user_agent TEXT,
                        message TEXT,
                        details TEXT,
                        success BOOLEAN,
                        encrypted BOOLEAN DEFAULT FALSE,
                        checksum TEXT
                    )
                """)
                
                # Indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)")
                
                # Alert history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS security_alerts (
                        alert_id TEXT PRIMARY KEY,
                        timestamp DATETIME,
                        alert_type TEXT,
                        severity TEXT,
                        message TEXT,
                        event_count INTEGER,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at DATETIME
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.datetime.now().isoformat()
        return hashlib.sha256(f"{timestamp}{os.urandom(16).hex()}".encode()).hexdigest()[:16]
    
    def _calculate_checksum(self, event_data: str) -> str:
        """Calculate integrity checksum for event data"""
        return hashlib.sha256(event_data.encode()).hexdigest()
    
    def log_event(self, 
                  event_type: EventType,
                  message: str,
                  severity: SeverityLevel = SeverityLevel.INFO,
                  user_id: str = "system",
                  session_id: Optional[str] = None,
                  source_ip: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  success: bool = True) -> str:
        """
        Log an audit event
        
        Args:
            event_type: Type of event
            message: Human-readable message
            severity: Event severity
            user_id: User identifier
            session_id: Session identifier
            source_ip: Source IP address
            user_agent: User agent string
            details: Additional event details
            success: Whether the operation was successful
            
        Returns:
            Event ID
        """
        try:
            event_id = self._generate_event_id()
            timestamp = datetime.datetime.now()
            
            event = AuditEvent(
                event_id=event_id,
                timestamp=timestamp,
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                session_id=session_id,
                source_ip=source_ip,
                user_agent=user_agent,
                message=message,
                details=details or {},
                success=success
            )\n            \n            # Add to queue for processing\n            if not self.event_queue.full():\n                self.event_queue.put(event)\n            else:\n                logger.warning(\"Audit event queue full, dropping event\")\n            \n            # Add to recent events cache for pattern detection\n            with self.cache_lock:\n                self.recent_events.append(event)\n                # Keep only last 1000 events\n                if len(self.recent_events) > 1000:\n                    self.recent_events = self.recent_events[-1000:]\n            \n            return event_id\n            \n        except Exception as e:\n            logger.error(f\"Failed to log audit event: {e}\")\n            return \"\"\n    \n    def start_processing(self):\n        \"\"\"Start background event processing\"\"\"\n        if self.processing_thread is None or not self.processing_thread.is_alive():\n            self.processing_thread = threading.Thread(target=self._process_events, daemon=True)\n            self.processing_thread.start()\n    \n    def _process_events(self):\n        \"\"\"Background thread for processing audit events\"\"\"\n        while self.running:\n            try:\n                # Get event from queue with timeout\n                event = self.event_queue.get(timeout=1.0)\n                \n                # Store in database\n                self._store_event(event)\n                \n                # Check for security patterns\n                self._check_security_patterns(event)\n                \n                # Write to file log\n                self._write_file_log(event)\n                \n                self.event_queue.task_done()\n                \n            except Empty:\n                continue\n            except Exception as e:\n                logger.error(f\"Error processing audit event: {e}\")\n    \n    def _store_event(self, event: AuditEvent):\n        \"\"\"Store event in database\"\"\"\n        try:\n            with sqlite3.connect(self.db_path) as conn:\n                cursor = conn.cursor()\n                \n                # Prepare event data\n                details_json = json.dumps(event.details)\n                event_data = f\"{event.event_id}{event.timestamp.isoformat()}{event.event_type.value}{event.message}{details_json}\"\n                checksum = self._calculate_checksum(event_data)\n                \n                # Encrypt sensitive details if enabled\n                encrypted = False\n                if self.encrypt_logs and event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:\n                    try:\n                        from core.encryption import encrypt_sensitive_data\n                        details_json = encrypt_sensitive_data(details_json, \"audit_log\")\n                        encrypted = True\n                    except Exception as e:\n                        logger.warning(f\"Failed to encrypt audit log: {e}\")\n                \n                cursor.execute(\"\"\"\n                    INSERT INTO audit_events \n                    (event_id, timestamp, event_type, severity, user_id, session_id, \n                     source_ip, user_agent, message, details, success, encrypted, checksum)\n                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n                \"\"\", (\n                    event.event_id,\n                    event.timestamp,\n                    event.event_type.value,\n                    event.severity.value,\n                    event.user_id,\n                    event.session_id,\n                    event.source_ip,\n                    event.user_agent,\n                    event.message,\n                    details_json,\n                    event.success,\n                    encrypted,\n                    checksum\n                ))\n                \n                conn.commit()\n                \n        except Exception as e:\n            logger.error(f\"Failed to store audit event: {e}\")\n    \n    def _write_file_log(self, event: AuditEvent):\n        \"\"\"Write event to daily log file\"\"\"\n        try:\n            date_str = event.timestamp.strftime(\"%Y-%m-%d\")\n            log_file = self.log_dir / f\"audit_{date_str}.log\"\n            \n            log_entry = {\n                \"timestamp\": event.timestamp.isoformat(),\n                \"event_id\": event.event_id,\n                \"type\": event.event_type.value,\n                \"severity\": event.severity.value,\n                \"user\": event.user_id,\n                \"session\": event.session_id,\n                \"ip\": event.source_ip,\n                \"message\": event.message,\n                \"success\": event.success\n            }\n            \n            with open(log_file, \"a\", encoding=\"utf-8\") as f:\n                f.write(json.dumps(log_entry) + \"\\n\")\n                \n        except Exception as e:\n            logger.error(f\"Failed to write file log: {e}\")\n    \n    def _check_security_patterns(self, event: AuditEvent):\n        \"\"\"Check for security patterns and generate alerts\"\"\"\n        try:\n            # Count recent events of same type\n            current_time = datetime.datetime.now()\n            recent_window = current_time - datetime.timedelta(minutes=5)\n            \n            with self.cache_lock:\n                recent_count = sum(1 for e in self.recent_events \n                                 if e.event_type == event.event_type and \n                                    e.timestamp >= recent_window)\n            \n            # Check if threshold exceeded\n            threshold = self.alert_thresholds.get(event.event_type)\n            if threshold and recent_count >= threshold:\n                self._generate_security_alert(\n                    event.event_type,\n                    f\"Threshold exceeded: {recent_count} {event.event_type.value} events in 5 minutes\",\n                    SeverityLevel.HIGH,\n                    recent_count\n                )\n                \n        except Exception as e:\n            logger.error(f\"Failed to check security patterns: {e}\")\n    \n    def _generate_security_alert(self, event_type: EventType, message: str, \n                               severity: SeverityLevel, event_count: int):\n        \"\"\"Generate security alert\"\"\"\n        try:\n            alert_id = self._generate_event_id()\n            timestamp = datetime.datetime.now()\n            \n            with sqlite3.connect(self.db_path) as conn:\n                cursor = conn.cursor()\n                cursor.execute(\"\"\"\n                    INSERT INTO security_alerts \n                    (alert_id, timestamp, alert_type, severity, message, event_count)\n                    VALUES (?, ?, ?, ?, ?, ?)\n                \"\"\", (\n                    alert_id,\n                    timestamp,\n                    event_type.value,\n                    severity.value,\n                    message,\n                    event_count\n                ))\n                conn.commit()\n            \n            # Log the alert as an audit event\n            self.log_event(\n                EventType.SECURITY_SUSPICIOUS_ACTIVITY,\n                f\"SECURITY ALERT: {message}\",\n                severity,\n                details={\"alert_id\": alert_id, \"event_count\": event_count}\n            )\n            \n            logger.warning(f\"SECURITY ALERT: {message}\")\n            \n        except Exception as e:\n            logger.error(f\"Failed to generate security alert: {e}\")\n    \n    def query_events(self, \n                    start_time: Optional[datetime.datetime] = None,\n                    end_time: Optional[datetime.datetime] = None,\n                    event_types: Optional[List[EventType]] = None,\n                    user_id: Optional[str] = None,\n                    severity: Optional[SeverityLevel] = None,\n                    limit: int = 1000) -> List[Dict[str, Any]]:\n        \"\"\"Query audit events with filters\"\"\"\n        try:\n            with sqlite3.connect(self.db_path) as conn:\n                conn.row_factory = sqlite3.Row\n                cursor = conn.cursor()\n                \n                # Build query\n                conditions = []\n                params = []\n                \n                if start_time:\n                    conditions.append(\"timestamp >= ?\")\n                    params.append(start_time)\n                \n                if end_time:\n                    conditions.append(\"timestamp <= ?\")\n                    params.append(end_time)\n                \n                if event_types:\n                    type_placeholders = \",\".join([\"?\"] * len(event_types))\n                    conditions.append(f\"event_type IN ({type_placeholders})\")\n                    params.extend([et.value for et in event_types])\n                \n                if user_id:\n                    conditions.append(\"user_id = ?\")\n                    params.append(user_id)\n                \n                if severity:\n                    conditions.append(\"severity = ?\")\n                    params.append(severity.value)\n                \n                where_clause = \" AND \".join(conditions) if conditions else \"1=1\"\n                query = f\"\"\"\n                    SELECT * FROM audit_events \n                    WHERE {where_clause} \n                    ORDER BY timestamp DESC \n                    LIMIT ?\n                \"\"\"\n                params.append(limit)\n                \n                cursor.execute(query, params)\n                rows = cursor.fetchall()\n                \n                # Convert to dictionaries and decrypt if needed\n                events = []\n                for row in rows:\n                    event_dict = dict(row)\n                    \n                    # Decrypt details if encrypted\n                    if event_dict.get('encrypted'):\n                        try:\n                            from core.encryption import decrypt_sensitive_data\n                            event_dict['details'] = decrypt_sensitive_data(\n                                event_dict['details'], \"audit_log\"\n                            )\n                        except Exception as e:\n                            logger.warning(f\"Failed to decrypt audit log: {e}\")\n                    \n                    events.append(event_dict)\n                \n                return events\n                \n        except Exception as e:\n            logger.error(f\"Failed to query audit events: {e}\")\n            return []\n    \n    def get_security_alerts(self, resolved: Optional[bool] = None) -> List[Dict[str, Any]]:\n        \"\"\"Get security alerts\"\"\"\n        try:\n            with sqlite3.connect(self.db_path) as conn:\n                conn.row_factory = sqlite3.Row\n                cursor = conn.cursor()\n                \n                if resolved is not None:\n                    cursor.execute(\"\"\"\n                        SELECT * FROM security_alerts \n                        WHERE resolved = ? \n                        ORDER BY timestamp DESC\n                    \"\"\", (resolved,))\n                else:\n                    cursor.execute(\"\"\"\n                        SELECT * FROM security_alerts \n                        ORDER BY timestamp DESC\n                    \"\"\")\n                \n                return [dict(row) for row in cursor.fetchall()]\n                \n        except Exception as e:\n            logger.error(f\"Failed to get security alerts: {e}\")\n            return []\n    \n    def generate_compliance_report(self, start_date: str, end_date: str) -> Dict[str, Any]:\n        \"\"\"Generate compliance report for given date range\"\"\"\n        try:\n            start_dt = datetime.datetime.fromisoformat(start_date)\n            end_dt = datetime.datetime.fromisoformat(end_date)\n            \n            events = self.query_events(start_time=start_dt, end_time=end_dt, limit=100000)\n            \n            # Analyze events\n            report = {\n                \"period\": {\"start\": start_date, \"end\": end_date},\n                \"total_events\": len(events),\n                \"event_breakdown\": {},\n                \"severity_breakdown\": {},\n                \"auth_events\": 0,\n                \"failed_auth_attempts\": 0,\n                \"system_events\": 0,\n                \"api_events\": 0,\n                \"security_events\": 0,\n                \"unique_users\": set(),\n                \"alerts_generated\": len(self.get_security_alerts())\n            }\n            \n            for event in events:\n                event_type = event['event_type']\n                severity = event['severity']\n                \n                # Count by type\n                report['event_breakdown'][event_type] = report['event_breakdown'].get(event_type, 0) + 1\n                \n                # Count by severity\n                report['severity_breakdown'][severity] = report['severity_breakdown'].get(severity, 0) + 1\n                \n                # Category counts\n                if event_type.startswith('auth.'):\n                    report['auth_events'] += 1\n                    if event_type == 'auth.login.failure':\n                        report['failed_auth_attempts'] += 1\n                elif event_type.startswith('system.'):\n                    report['system_events'] += 1\n                elif event_type.startswith('api.'):\n                    report['api_events'] += 1\n                elif event_type.startswith('security.'):\n                    report['security_events'] += 1\n                \n                # Unique users\n                if event['user_id']:\n                    report['unique_users'].add(event['user_id'])\n            \n            report['unique_users'] = len(report['unique_users'])\n            \n            return report\n            \n        except Exception as e:\n            logger.error(f\"Failed to generate compliance report: {e}\")\n            return {}\n    \n    def cleanup_old_logs(self, days_to_keep: int = 90):\n        \"\"\"Clean up old audit logs\"\"\"\n        try:\n            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)\n            \n            with sqlite3.connect(self.db_path) as conn:\n                cursor = conn.cursor()\n                \n                # Archive old events to file before deletion\n                cursor.execute(\n                    \"SELECT * FROM audit_events WHERE timestamp < ?\", \n                    (cutoff_date,)\n                )\n                \n                old_events = cursor.fetchall()\n                if old_events:\n                    archive_file = self.log_dir / f\"archived_events_{cutoff_date.strftime('%Y%m%d')}.json\"\n                    with open(archive_file, 'w') as f:\n                        json.dump([dict(zip([col[0] for col in cursor.description], row)) \n                                  for row in old_events], f, default=str)\n                \n                # Delete old events\n                cursor.execute(\"DELETE FROM audit_events WHERE timestamp < ?\", (cutoff_date,))\n                \n                # Delete old alerts\n                cursor.execute(\n                    \"DELETE FROM security_alerts WHERE timestamp < ? AND resolved = ?\", \n                    (cutoff_date, True)\n                )\n                \n                conn.commit()\n                \n                logger.info(f\"Cleaned up {len(old_events)} old audit events\")\n                \n        except Exception as e:\n            logger.error(f\"Failed to cleanup old logs: {e}\")\n    \n    def stop(self):\n        \"\"\"Stop audit logging\"\"\"\n        self.running = False\n        if self.processing_thread:\n            self.processing_thread.join(timeout=5.0)\n\n\n# Global audit logger instance\n_audit_logger = None\n\ndef get_audit_logger() -> AuditLogger:\n    \"\"\"Get global audit logger instance\"\"\"\n    global _audit_logger\n    if _audit_logger is None:\n        _audit_logger = AuditLogger()\n    return _audit_logger\n\n\n# Convenience functions for common audit events\ndef audit_auth_success(user_id: str, session_id: str, source_ip: str = None):\n    \"\"\"Log successful authentication\"\"\"\n    get_audit_logger().log_event(\n        EventType.AUTH_LOGIN_SUCCESS,\n        f\"User {user_id} authenticated successfully\",\n        SeverityLevel.INFO,\n        user_id=user_id,\n        session_id=session_id,\n        source_ip=source_ip\n    )\n\ndef audit_auth_failure(user_id: str, reason: str, source_ip: str = None):\n    \"\"\"Log failed authentication\"\"\"\n    get_audit_logger().log_event(\n        EventType.AUTH_LOGIN_FAILURE,\n        f\"Authentication failed for {user_id}: {reason}\",\n        SeverityLevel.MEDIUM,\n        user_id=user_id,\n        source_ip=source_ip,\n        success=False\n    )\n\ndef audit_system_command(command: str, user_id: str, success: bool = True):\n    \"\"\"Log system command execution\"\"\"\n    get_audit_logger().log_event(\n        EventType.SYSTEM_COMMAND_EXEC,\n        f\"System command executed: {command}\",\n        SeverityLevel.MEDIUM,\n        user_id=user_id,\n        success=success,\n        details={\"command\": command}\n    )\n\ndef audit_api_request(endpoint: str, user_id: str, source_ip: str = None, success: bool = True):\n    \"\"\"Log API request\"\"\"\n    get_audit_logger().log_event(\n        EventType.API_REQUEST,\n        f\"API request to {endpoint}\",\n        SeverityLevel.LOW,\n        user_id=user_id,\n        source_ip=source_ip,\n        success=success,\n        details={\"endpoint\": endpoint}\n    )\n\ndef audit_data_access(data_type: str, user_id: str, operation: str = \"read\"):\n    \"\"\"Log data access\"\"\"\n    get_audit_logger().log_event(\n        EventType.DATA_ACCESS,\n        f\"Data access: {operation} {data_type}\",\n        SeverityLevel.LOW,\n        user_id=user_id,\n        details={\"data_type\": data_type, \"operation\": operation}\n    )\n\ndef audit_security_event(message: str, severity: SeverityLevel = SeverityLevel.HIGH, \n                        user_id: str = \"system\", details: Dict[str, Any] = None):\n    \"\"\"Log security event\"\"\"\n    get_audit_logger().log_event(\n        EventType.SECURITY_SUSPICIOUS_ACTIVITY,\n        message,\n        severity,\n        user_id=user_id,\n        details=details or {}\n    )\n\n\nif __name__ == \"__main__\":\n    # Test audit logging\n    print(\"Testing audit logging system...\")\n    \n    audit_logger = AuditLogger()\n    \n    # Test various events\n    audit_auth_success(\"test_user\", \"session_123\", \"127.0.0.1\")\n    audit_auth_failure(\"test_user\", \"Invalid PIN\", \"127.0.0.1\")\n    audit_system_command(\"ls -la\", \"test_user\")\n    audit_api_request(\"/api/chat\", \"test_user\", \"127.0.0.1\")\n    \n    # Wait for processing\n    import time\n    time.sleep(2)\n    \n    # Query events\n    events = audit_logger.query_events(limit=10)\n    print(f\"Found {len(events)} events\")\n    \n    # Generate report\n    report = audit_logger.generate_compliance_report(\n        (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat(),\n        datetime.datetime.now().isoformat()\n    )\n    print(f\"Compliance report: {report}\")\n    \n    audit_logger.stop()\n    print(\"✅ Audit logging test completed!\")