---
description: Create high-level feature request for current sprint (project)
argument-hint: [feature-description]
---

# Create Feature Request

## Workflow

### Step 1: Get current sprint

Read `CLAUDE.md` to find the current sprint name/number.

### Step 2: Read feature description

Feature description from: $ARGUMENTS

### Step 3: Find or create feature request file

File location: `PRPs/feature-requests/sprint-{sprint-no}-feature-requests.md`

If file doesn't exist, create it with basic structure.

### Step 4: Add feature request

Append the feature to the markdown file:

**CRITICAL GUIDELINES - READ BEFORE WRITING:**

✅ **DO INCLUDE:**
- WHAT the feature does (user-facing functionality)
- WHY it's needed (business value, user pain point)
- WHAT the expected behavior is (outcomes, not mechanics)
- WHICH systems/integrations are affected (high-level)
- WHAT success looks like (measurable outcomes)

❌ **DO NOT INCLUDE:**
- Code snippets or file paths
- Specific function/class/variable names
- Database queries or schema details
- API endpoint implementations
- Technical architecture decisions
- Specific algorithms or data structures
- Line numbers or code references
- Frontend component hierarchies
- Detailed state management approaches

**Examples:**

❌ BAD (Too Technical):
```markdown
**Technical Context:**
```typescript
// Frontend (projects.ts:75-78) - Missing request body
restore: async (projectId: number) => {
  const response = await apiClient.post(`/api/projects/${projectId}/restore`)
}
```
Backend expects RestoreProjectRequest with to_status_id field.
```

✅ GOOD (High-Level):
```markdown
**Problem:**
Users cannot restore archived projects back to the active workflow. When clicking "Restore" on an archived project, nothing happens and the project remains archived.

**Expected Behavior:**
Clicking "Restore" should move the project back to the Kanban board with its previous status, and log the restoration in the activity history.
```

---

**Standard Format:**
```markdown
## Feature: {Short Title}

**Description:**
{What this feature does and why it's needed - focus on user value}

**Key Requirements:**
- {User-facing requirement 1}
- {User-facing requirement 2}
- {Integration requirement - system-level only}

**Integration Points:**
- {System/service names only - no file paths}

**Success Criteria:**
- {How we know it works - user perspective}
```
