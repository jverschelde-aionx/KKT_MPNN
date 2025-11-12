# Coding Conventions

## Import Statements

**All imports must be at the top of the file. Never use inline imports.**

**DO:**
- ✅ Place all imports at the top of the file (after module docstring)
- ✅ Organize imports alphabetically within their groups
- ✅ Group imports: standard library, third-party, local imports


**DON'T:**
- ❌ Put imports inside functions or methods
- ❌ Use inline imports for lazy loading
- ❌ Import within try/except blocks inside functions

**Example (Good):**
```python
"""Module docstring."""

import os
import sys
from typing import Optional

import ezdxf
import requests

from shared.database import DatabaseClient
```

**Example (Bad):**
```python
def process_file():
    import ezdxf  # ❌ Never do this
    try:
        import optional_lib  # ❌ Wrong location
    except ImportError:
        pass
```

## Exception Handling

**Only catch exceptions when you can actually fix the problem.**

Application Insights automatically captures and logs all unhandled exceptions. Therefore:

**DO catch exceptions when:**
- ✅ Implementing retry logic (transient failures)
- ✅ Providing fallback behavior (graceful degradation)
- ✅ Cleaning up resources (finally blocks)
- ✅ Continuing execution despite non-critical failures
- ✅ Transforming exceptions (e.g., wrapping in domain-specific exceptions)

**DON'T catch exceptions when:**
- ❌ Just to log and re-raise - Application Insights already does this
- ❌ Just to add context - Application Insights captures stack traces
- ❌ You can't actually recover or provide alternative behavior
- ❌ Just to "handle" them without fixing anything
- ❌ To check if dependencies are installed