# UI Components & Design System

## 🎨 Design Philosophy

**Unique Visual Identity**: Data-dense, technical aesthetic with emerald green accent colors, dark theme, and glassmorphic elements.

### Color Palette

```css
Primary Colors:
  Emerald: #10b981 → Main brand color (buttons, accents)
  Emerald Dark: #059669 → Hover states, gradients

Secondary Colors:
  Blue: #3b82f6 → Links, secondary actions
  Green Success: #22c55e → Validation success
  Red Error: #ef4444 → Error states
  Orange Warning: #f59e0b → Warnings

Background Layers:
  Dark Base: #0f172a → Main background
  Card Dark: #1e293b → Elevated surfaces
  Card Light: #334155 → Hover states

Text:
  Primary: #e2e8f0 → Main text
  Secondary: #94a3b8 → Labels, metadata
```

## 🧩 Component Library

### 1. Header Hero

```
╔═══════════════════════════════════════════════════════╗
║ 🗄️ Text-to-SQL System                                ║
║                                                       ║
║ Transform natural language into executable SQL       ║
║ queries using AI                                      ║
╚═══════════════════════════════════════════════════════╝
```

**Features:**
- Linear gradient background (emerald → dark emerald)
- Subtle box shadow with emerald glow
- Large bold title with text shadow
- Descriptive subtitle

**CSS:**
```css
background: linear-gradient(135deg, #10b981 0%, #059669 100%)
box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2)
border-radius: 12px
padding: 2rem
```

---

### 2. Status Badges

```
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│ ✓ Valid SQL │  │ ✗ Invalid SQL│  │ ⚠ Warning    │
└─────────────┘  └──────────────┘  └──────────────┘
  (Green)          (Red)             (Orange)
```

**Usage:**
- SQL validation status
- Model readiness indicator
- Query execution results

**CSS:**
```css
display: inline-block
padding: 0.4rem 1rem
border-radius: 20px (pill shape)
font-weight: 600
```

---

### 3. Metric Cards

```
┌─────────────────────────────┐
│ CONFIDENCE                  │
│                             │
│        95.2%                │
│                             │
└─────────────────────────────┘
    (Gradient background)
    (Hover: lifts up 2px)
```

**Features:**
- Dark gradient background
- Semi-transparent border
- Hover animation (translateY -2px)
- Large emphasized value
- Uppercase label

**CSS:**
```css
background: linear-gradient(135deg, #1e293b 0%, #334155 100%)
border: 1px solid rgba(255, 255, 255, 0.1)
border-radius: 12px
transition: transform 0.2s

:hover {
  transform: translateY(-2px)
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15)
}
```

---

### 4. Example Query Cards

```
┌─────────────────────────────────────────────────┐
│ 📝 Show all students with GPA above 3.5         │
└─────────────────────────────────────────────────┘
  (Left emerald border)
  (Hover: slides right 4px)
```

**Features:**
- Dark card background
- Left accent border (4px emerald)
- Hover animation (translateX 4px)
- Cursor pointer for interactivity

**CSS:**
```css
background: #1e293b
border-left: 4px solid #10b981
border-radius: 8px
transition: all 0.2s

:hover {
  background: #334155
  border-left-color: #22c55e
  transform: translateX(4px)
}
```

---

### 5. SQL Code Block

```sql
┌─────────────────────────────────────────┐
│  1  SELECT *                            │
│  2  FROM students                       │
│  3  WHERE gpa > 3.5                     │
└─────────────────────────────────────────┘
  (Dark background with syntax highlighting)
```

**Features:**
- Dark background (#0f172a)
- Line numbers enabled
- SQL syntax highlighting (Streamlit native)
- Semi-transparent border
- Subtle shadow

**CSS:**
```css
background: #0f172a
border: 1px solid #334155
border-radius: 8px
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2)
```

---

### 6. Primary Button

```
┌────────────────────────┐
│  🔄 Generate SQL       │
└────────────────────────┘
  (Gradient emerald background)
  (Hover: lifts up with glow)
```

**Features:**
- Gradient background (emerald → dark emerald)
- White text with medium weight
- Emerald glow shadow
- Hover animation (translateY -2px)
- Rounded corners

**CSS:**
```css
background: linear-gradient(135deg, #10b981 0%, #059669 100%)
color: white
border: none
padding: 0.75rem 2rem
border-radius: 8px
font-weight: 600
box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3)

:hover {
  transform: translateY(-2px)
  box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4)
}
```

---

### 7. Schema Expander

```
📋 View Schema Details ▼

┌─────────────────────────────────────────┐
│ students                                │
│   id, name, age, gpa, major, ...        │
│ ─────────────────────────────────────── │
│ professors                              │
│   id, name, department, salary, ...     │
└─────────────────────────────────────────┘
```

**Features:**
- Collapsible accordion
- Collapsed by default
- Bold table names
- Inline code styling for columns
- Dividers between tables

---

### 8. Results Dataframe

```
┌───────────────────────────────────────────────────┐
│ id │ name          │ age │ gpa  │ major          │
├────┼───────────────┼─────┼──────┼────────────────┤
│ 1  │ Alice Johnson │ 20  │ 3.8  │ Comp Sci       │
│ 2  │ Bob Smith     │ 21  │ 3.9  │ Mathematics    │
│ 3  │ Carol White   │ 22  │ 3.7  │ Physics        │
└────┴───────────────┴─────┴──────┴────────────────┘
  (Streamlit native dataframe with custom styling)
```

**Features:**
- Full width container
- Hidden index
- Sortable columns (Streamlit native)
- Rounded corners with shadow
- Hover row highlighting

---

### 9. Metric Display (Execution Stats)

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Rows        │  │ Columns     │  │ Exec Time   │
│   143       │  │     5       │  │  0.042s     │
└─────────────┘  └─────────────┘  └─────────────┘
  (Streamlit native metric components)
```

**Features:**
- Clean metric cards
- Large emphasized value
- Small label text
- Responsive grid layout

---

### 10. Download Buttons

```
┌──────────────────────┐  ┌──────────────────────┐
│ 📥 Download as CSV   │  │ 📥 Download as JSON  │
└──────────────────────┘  └──────────────────────┘
  (Secondary button style)
```

**Features:**
- Icon + text label
- Full width in container
- Timestamped filenames
- Streamlit native download

---

## 🎭 Animations & Transitions

### Hover Effects

```css
/* Metric Cards */
transform: translateY(-2px)
box-shadow: enhanced

/* Example Queries */
transform: translateX(4px)
background: lighter
border-color: brighter

/* Buttons */
transform: translateY(-2px)
box-shadow: larger glow
```

### Loading States

```
⟳ (Rotating spinner)
  or
▓▓▓▓▓▓░░░░ (Progress bar)
```

**Streamlit native spinners with emerald color override:**
```css
.stSpinner > div {
  border-color: #10b981 transparent transparent transparent
}
```

### Transitions

All interactive elements: `transition: all 0.2s`
- Smooth hover effects
- Button press feedback
- Color changes
- Shadow adjustments

---

## 📐 Layout System

### Page Layout

```
┌─────────────────────────────────────────────────────────┐
│                      [HEADER HERO]                      │
├─────────────┬───────────────────────────────────────────┤
│             │                                           │
│   SIDEBAR   │              MAIN CONTENT                 │
│             │                                           │
│  • Config   │  [Schema] ─────────── [Stats]            │
│  • Model    │                                           │
│  • History  │  [Example Queries]                        │
│             │                                           │
│             │  [Natural Language Input]                 │
│             │                                           │
│             │  [Generated SQL]                          │
│             │                                           │
│             │  [Execution Results]                      │
│             │                                           │
├─────────────┴───────────────────────────────────────────┤
│                       [FOOTER]                          │
└─────────────────────────────────────────────────────────┘
```

### Column Ratios

```python
# Main content split
col1, col2 = st.columns([2, 1])  # 66% / 33%

# Metric displays
cols = st.columns(4)  # Equal width

# Download buttons
cols = st.columns(2)  # 50% / 50%
```

### Spacing

```css
Section breaks: st.markdown("---")
Card padding: 1.5rem
Button padding: 0.75rem 2rem
Page margin: 2rem
Gap between elements: 1rem
```

---

## 🎯 Responsive Breakpoints

```css
Desktop (1920px+):
  - Wide layout enabled
  - Side-by-side columns
  - Full metric grid (4 columns)

Laptop (1366px+):
  - Wide layout maintained
  - Compressed sidebar
  - Metric grid intact

Tablet (768px+):
  - Sidebar collapsible
  - Columns stack vertically
  - Metric grid 2x2

Mobile (375px+):
  - Single column layout
  - Sidebar hidden by default
  - Stacked metrics
  - Full-width buttons
```

---

## 🌈 Visual Hierarchy

### Priority Levels

**Primary (Highest Attention):**
- Generate SQL button (emerald gradient, glow)
- Header hero (large, colorful)
- Status badges (bright colors)

**Secondary:**
- Example query cards (hover highlight)
- Metric cards (subtle gradient)
- SQL code blocks (emphasized)

**Tertiary:**
- Schema expander (collapsed)
- Footer text (muted)
- Labels (small, gray)

### Typography Scale

```
Hero Title: 2.5rem / 700 weight
Section Headers: 1.5rem / 600 weight
Body Text: 1rem / 400 weight
Labels: 0.9rem / 400 weight / uppercase
Code: monospace / 0.95rem
```

---

## 💡 Unique Design Elements

### 1. Shimmer Effect (Button Hover)

```css
/* Animated gradient overlay on hover */
position: absolute
background: linear-gradient(
  to right,
  transparent,
  rgba(255, 255, 255, 0.1),
  transparent
)
animation: shimmer 0.7s
```

### 2. Glassmorphism (Future Enhancement)

```css
/* Semi-transparent cards with backdrop blur */
background: rgba(30, 41, 59, 0.6)
backdrop-filter: blur(10px)
border: 1px solid rgba(255, 255, 255, 0.1)
```

### 3. Gradient Borders (Alternative Design)

```css
/* Rainbow border for premium feel */
border: 2px solid transparent
background-clip: padding-box
background-image: linear-gradient(135deg, #10b981, #3b82f6, #8b5cf6)
```

### 4. Glow Effects

```css
/* Emerald glow on interactive elements */
box-shadow: 0 0 20px rgba(16, 185, 129, 0.4)
filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.3))
```

---

## 🚀 Performance Optimizations

### Cached Resources

```python
@st.cache_resource  # Model loading
@st.cache_data      # Static data (schemas, examples)
```

### Session State

```python
st.session_state.query_history      # Query tracking
st.session_state.generated_sql      # Cached generation
st.session_state.execution_results  # Cached results
```

### Lazy Loading

- Sidebar collapsed initially → Load on expand
- Schema details hidden → Load on expand
- Model loading on demand → Not at startup

---

## 📊 Component Usage Guide

### When to Use Each Component

| Component | Use Case |
|-----------|----------|
| Status Badge | Binary states (success/error/warning) |
| Metric Card | Numerical KPIs (confidence, time, count) |
| Example Card | Clickable templates or suggestions |
| Code Block | Displaying generated SQL or logs |
| Dataframe | Tabular query results |
| Expander | Hiding details until needed |
| Download Button | Exporting data (CSV/JSON) |

---

## 🎨 Customization Quick Reference

**Change Primary Color:**
```css
--primary: #10b981 → YOUR_COLOR
```

**Change Background:**
```css
--bg-dark: #0f172a → YOUR_BG
```

**Adjust Border Radius:**
```css
border-radius: 12px → YOUR_RADIUS
```

**Modify Shadow Intensity:**
```css
box-shadow: 0 4px 12px rgba(0,0,0,0.2) → YOUR_SHADOW
```

---

**Design System v1.0 | Built for Text-to-SQL Demo**
