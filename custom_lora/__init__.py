# Copyright 2025-present YourName.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import sys
from peft.utils import PeftType


def _patch_peft():
    """
    Dynamically add VAELORA to PeftType enum and register the method.
    """

    # 1. Create a new enum with the additional type
    existing_members = {member.name: member.value for member in PeftType}
    if not hasattr(PeftType, 'VAELORA'):
        existing_members['VAELORA'] = 'VAELORA'
    
    # Create new enum with str mixin
    NewPeftType = enum.Enum(
        'PeftType',  # 使用固定名称而不是 PeftType.__name__
        existing_members, 
        type=str
    )
    
    # 2. 更彻底的模块替换策略
    original_peft_type = PeftType
    
    # 替换所有已加载模块中的 PeftType 引用
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue
        
        # 检查所有可能包含 PeftType 的模块
        if any(module_name.startswith(prefix) for prefix in ['peft', 'custom_lora']):
            if hasattr(module, 'PeftType') and getattr(module, 'PeftType') is original_peft_type:
                setattr(module, 'PeftType', NewPeftType)
    
    # 3. 特别处理 peft.utils 模块（这是最常见的导入来源）
    import peft.utils
    peft.utils.PeftType = NewPeftType
    
    # 4. 更新当前模块的全局命名空间
    globals()['PeftType'] = NewPeftType
    
    return NewPeftType


# 立即执行补丁，在任何其他导入之前
_patched_PeftType = _patch_peft()
print("run here in peft init")


from .vaelora import (
    VAELoRAConfig,
    VAELoRAModel,
)


__all__ = [
    "VAELoRAConfig",
    "VAELoRAModel",
]
