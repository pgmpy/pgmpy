# Feature Enhancement for Issue #1000
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ConfigManager:
    config: Dict[str, any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
    
    def get(self, key: str, default: Optional[any] = None) -> any:
        return self.config.get(key, default)
    
    def set(self, key: str, value: any) -> None:
        self.config[key] = value
    
    def save(self) -> Dict[str, any]:
        return self.config.copy()

# 测试
config = ConfigManager()
config.set("test", "value")
assert config.get("test") == "value"
assert config.save()["test"] == "value"
print("Feature enhancement tests passed!")
