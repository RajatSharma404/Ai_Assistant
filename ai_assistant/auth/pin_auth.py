"""
PIN Authentication System for YourDaddy AI Assistant

This module handles PIN-based authentication for the AI Assistant,
replacing traditional login pages with a simple PIN prompt at startup.
"""

import os
import sys
import hashlib
import secrets
import getpass
from pathlib import Path
from typing import Optional, Tuple
import json

try:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class PINAuth:
    """Handle PIN-based authentication for the AI Assistant"""
    
    def __init__(self, config_file: str = "config/app_integration.env"):
        """
        Initialize PIN authentication system
        
        Args:
            config_file: Path to configuration file containing PIN hash
        """
        self.config_file = Path(config_file)
        self.user_settings_file = Path("config/user_settings.json")
        self.pin_hash = None
        self.salt = None
        self._load_pin_config()
    
    def _load_pin_config(self) -> None:
        """Load PIN configuration from environment file"""
        try:
            # Check if PIN is configured in environment file
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract PIN hash and salt from env file
                for line in content.split('\n'):
                    if line.startswith('PIN_HASH='):
                        self.pin_hash = line.split('=', 1)[1].strip()
                    elif line.startswith('PIN_SALT='):
                        self.salt = line.split('=', 1)[1].strip()
            
            # Fallback to user settings file
            if not self.pin_hash and self.user_settings_file.exists():
                try:
                    with open(self.user_settings_file, 'r', encoding='utf-8') as f:
                        settings = json.load(f)
                        
                    # Look for PIN in security settings
                    for setting_group in settings.get('settings', []):
                        if setting_group.get('title') == 'Security & Privacy':
                            pin_config = setting_group.get('pin_config', {})
                            self.pin_hash = pin_config.get('pin_hash')
                            self.salt = pin_config.get('salt')
                            break
                            
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Could not read PIN from user settings: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading PIN configuration: {e}")
    
    def _hash_pin(self, pin: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash a PIN with salt for secure storage
        
        Args:
            pin: The PIN to hash
            salt: Optional salt (generates new one if not provided)
            
        Returns:
            Tuple of (hashed_pin, salt)
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for secure hashing
        pin_bytes = pin.encode('utf-8')
        salt_bytes = salt.encode('utf-8')
        
        # 100,000 iterations for security
        hash_obj = hashlib.pbkdf2_hmac('sha256', pin_bytes, salt_bytes, 100000)
        hashed_pin = hash_obj.hex()
        
        return hashed_pin, salt
    
    def is_pin_configured(self) -> bool:
        """Check if a PIN is already configured"""
        return self.pin_hash is not None and self.salt is not None
    
    def setup_pin(self, pin: str) -> bool:
        """
        Set up a new PIN for the assistant
        
        Args:
            pin: The PIN to configure
            
        Returns:
            True if PIN was successfully configured, False otherwise
        """
        try:
            # Validate PIN
            if len(pin) < 4:
                print("❌ PIN must be at least 4 digits long")
                return False
                
            if not pin.isdigit():
                print("❌ PIN must contain only numbers")
                return False
            
            # Hash the PIN
            hashed_pin, salt = self._hash_pin(pin)
            
            # Save to environment file
            self._save_pin_to_env(hashed_pin, salt)
            
            # Update instance variables
            self.pin_hash = hashed_pin
            self.salt = salt
            
            logger.info("PIN configured successfully")
            print("✅ PIN configured successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up PIN: {e}")
            print(f"❌ Error setting up PIN: {e}")
            return False
    
    def _save_pin_to_env(self, hashed_pin: str, salt: str) -> None:
        """Save PIN hash and salt to environment file"""
        try:
            # Create config directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Read existing content
            existing_content = ""
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
            
            # Remove existing PIN entries
            lines = []
            for line in existing_content.split('\n'):
                if not (line.startswith('PIN_HASH=') or line.startswith('PIN_SALT=')):
                    lines.append(line)
            
            # Add PIN section if not present
            if not any('PIN AUTHENTICATION' in line for line in lines):
                lines.extend([
                    "",
                    "# =============================================================================",
                    "# PIN AUTHENTICATION",
                    "# =============================================================================",
                    ""
                ])
            
            # Add PIN configuration
            lines.extend([
                f"PIN_HASH={hashed_pin}",
                f"PIN_SALT={salt}"
            ])
            
            # Write back to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            logger.error(f"Error saving PIN to environment file: {e}")
            raise
    
    def verify_pin(self, pin: str) -> bool:
        """
        Verify a PIN against the stored hash
        
        Args:
            pin: The PIN to verify
            
        Returns:
            True if PIN is correct, False otherwise
        """
        try:
            if not self.is_pin_configured():
                logger.warning("No PIN configured")
                return False
            
            # Hash the provided PIN with stored salt
            hashed_input, _ = self._hash_pin(pin, self.salt)
            
            # Compare hashes
            is_valid = hashed_input == self.pin_hash
            
            # Log authentication attempt
            if is_valid:
                logger.info("Successful PIN authentication")
            else:
                logger.warning("Failed PIN authentication attempt")
                
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying PIN: {e}")
            return False
    
    def prompt_for_pin(self, max_attempts: int = 3) -> bool:
        """
        Prompt user for PIN with limited attempts
        
        Args:
            max_attempts: Maximum number of authentication attempts
            
        Returns:
            True if authentication successful, False otherwise
        """
        print("\n🔐 AI Assistant PIN Authentication")
        print("=" * 40)
        
        if not self.is_pin_configured():
            print("⚠️  No PIN configured. Setting up PIN authentication...")
            return self._setup_new_pin()
        
        for attempt in range(max_attempts):
            try:
                # Get PIN input (hidden)
                pin = getpass.getpass(f"Enter PIN ({attempt + 1}/{max_attempts}): ")
                
                if self.verify_pin(pin):
                    print("✅ Authentication successful!")
                    return True
                else:
                    remaining = max_attempts - attempt - 1
                    if remaining > 0:
                        print(f"❌ Invalid PIN. {remaining} attempts remaining.")
                    else:
                        print("❌ Maximum attempts exceeded. Access denied.")
                        
            except KeyboardInterrupt:
                print("\n❌ Authentication cancelled by user")
                return False
            except Exception as e:
                logger.error(f"Error during PIN prompt: {e}")
                print(f"❌ Authentication error: {e}")
                
        return False
    
    def _setup_new_pin(self) -> bool:
        """Set up a new PIN interactively"""
        try:
            print("\n📝 PIN Setup")
            print("-" * 20)
            print("PIN Requirements:")
            print("• At least 4 digits")
            print("• Numbers only")
            print("• Easy to remember but not obvious")
            print()
            
            while True:
                pin1 = getpass.getpass("Enter new PIN: ")
                
                if len(pin1) < 4:
                    print("❌ PIN must be at least 4 digits long")
                    continue
                    
                if not pin1.isdigit():
                    print("❌ PIN must contain only numbers")
                    continue
                
                pin2 = getpass.getpass("Confirm PIN: ")
                
                if pin1 != pin2:
                    print("❌ PINs do not match. Please try again.")
                    continue
                
                # Set up the PIN
                if self.setup_pin(pin1):
                    print("\n✅ PIN setup complete! Please restart the assistant.")
                    return True
                else:
                    print("❌ Failed to setup PIN. Please try again.")
                    
        except KeyboardInterrupt:
            print("\n❌ PIN setup cancelled")
            return False
        except Exception as e:
            logger.error(f"Error during PIN setup: {e}")
            print(f"❌ PIN setup error: {e}")
            return False
    
    def change_pin(self) -> bool:
        """Change the current PIN"""
        print("\n🔄 Change PIN")
        print("-" * 20)
        
        # Verify current PIN
        if self.is_pin_configured():
            current_pin = getpass.getpass("Enter current PIN: ")
            if not self.verify_pin(current_pin):
                print("❌ Current PIN is incorrect")
                return False
        
        # Set up new PIN
        return self._setup_new_pin()


def authenticate() -> bool:
    """
    Main authentication function to be called at startup
    
    Returns:
        True if authentication successful, False otherwise
    """
    auth = PINAuth()
    return auth.prompt_for_pin()


def setup_pin_cli():
    """CLI utility for PIN setup"""
    print("🔐 YourDaddy Assistant - PIN Setup Utility")
    print("=" * 50)
    
    auth = PINAuth()
    
    if auth.is_pin_configured():
        print("✅ PIN is already configured")
        choice = input("Do you want to change it? (y/N): ").lower()
        if choice == 'y':
            auth.change_pin()
    else:
        print("⚠️  No PIN configured. Setting up new PIN...")
        auth._setup_new_pin()


if __name__ == "__main__":
    # CLI entry point for PIN setup
    setup_pin_cli()