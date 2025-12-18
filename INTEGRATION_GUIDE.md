# Quick Integration Guide - Learning Systems

## Overview
This guide shows how to integrate the newly completed HIGH priority features into your application.

---

## 1. Training Data Feedback Loop Integration

### Backend (Already Complete ✅)
The feedback system is already integrated into [conversational_ai.py](ai_assistant/modules/conversational_ai.py). Every user interaction is automatically logged.

### Verify It's Working
```python
from ai_assistant.modules.conversational_ai import AdvancedConversationalAI

# Initialize
ai = AdvancedConversationalAI()

# Check feedback system is loaded
print(f"Feedback system active: {ai.feedback_system is not None}")

# Process a message
response = ai.process_message("open chrome")

# Check feedback database
import sqlite3
conn = sqlite3.connect('ai_assistant/databases/feedback_learning.db')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM interaction_logs")
count = cursor.fetchone()[0]
print(f"Total interactions logged: {count}")
```

### Access Training Data
```python
from ai_assistant.ai.advanced_feedback_learning import AdaptiveLearningEngine

engine = AdaptiveLearningEngine()

# Get recent interactions
interactions = engine.get_recent_interactions(limit=10)
for interaction in interactions:
    print(f"Prompt: {interaction['prompt']}")
    print(f"Response: {interaction['response']}")
    print(f"Context: {interaction['context']}")
    print("---")

# Get feedback entries
feedback = engine.get_feedback_summary()
print(f"Total feedback: {feedback}")
```

---

## 2. Knowledge Graph Visualization API Integration

### Start the API Server
```bash
# Option 1: Run standalone
cd f:\bn\assitant
python -m uvicorn ai_assistant.services.learning_api:router --reload --port 8000

# Option 2: Integrate into existing web backend (recommended)
# Edit modern_web_backend.py:
```

### Integrate into Web Backend
Add to [modern_web_backend.py](modern_web_backend.py):

```python
from fastapi import FastAPI
from ai_assistant.services.learning_api import router as learning_router

app = FastAPI()

# Include learning API router
app.include_router(learning_router)

# Your existing routes...
```

### Test API Endpoints
```bash
# Export knowledge graph
curl http://localhost:8000/api/learning/knowledge-graph/export

# Get graph statistics
curl http://localhost:8000/api/learning/knowledge-graph/stats

# Full dashboard
curl http://localhost:8000/api/learning/dashboard
```

---

## 3. Frontend Visualization (React Example)

### Install Dependencies
```bash
npm install d3 @types/d3
```

### Knowledge Graph Component
```typescript
// KnowledgeGraphVisualization.tsx
import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface GraphData {
  nodes: Array<{
    id: string;
    type: string;
    content: string;
    importance_score: number;
  }>;
  links: Array<{
    source: string;
    target: string;
    relationship_type: string;
    strength: number;
  }>;
  stats: {
    node_count: number;
    edge_count: number;
    node_types: Record<string, number>;
  };
}

export const KnowledgeGraphVisualization: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch graph data
    fetch('http://localhost:8000/api/learning/knowledge-graph/export')
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setGraphData(data.graph);
          setLoading(false);
        }
      });
  }, []);

  useEffect(() => {
    if (!graphData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const width = 1200;
    const height = 800;

    svg.attr('width', width).attr('height', height);

    // Clear previous content
    svg.selectAll('*').remove();

    // Create force simulation
    const simulation = d3.forceSimulation(graphData.nodes)
      .force('link', d3.forceLink(graphData.links).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2));

    // Draw links
    const link = svg.append('g')
      .selectAll('line')
      .data(graphData.links)
      .enter().append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d: any) => Math.sqrt(d.strength * 5));

    // Draw nodes
    const node = svg.append('g')
      .selectAll('circle')
      .data(graphData.nodes)
      .enter().append('circle')
      .attr('r', (d: any) => 5 + d.importance_score * 10)
      .attr('fill', (d: any) => {
        const colorMap: Record<string, string> = {
          person: '#3b82f6',
          skill: '#10b981',
          location: '#f59e0b',
          action: '#ef4444'
        };
        return colorMap[d.type] || '#6b7280';
      })
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // Add labels
    const label = svg.append('g')
      .selectAll('text')
      .data(graphData.nodes)
      .enter().append('text')
      .text((d: any) => d.content.substring(0, 20))
      .attr('font-size', 10)
      .attr('dx', 12)
      .attr('dy', 4);

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      label
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });

    // Drag functions
    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
  }, [graphData]);

  if (loading) {
    return <div>Loading knowledge graph...</div>;
  }

  return (
    <div className="knowledge-graph-container">
      <h2>Knowledge Graph Visualization</h2>
      {graphData && (
        <div className="stats">
          <p>Nodes: {graphData.stats.node_count}</p>
          <p>Edges: {graphData.stats.edge_count}</p>
          <div className="node-types">
            {Object.entries(graphData.stats.node_types).map(([type, count]) => (
              <span key={type}>{type}: {count} | </span>
            ))}
          </div>
        </div>
      )}
      <svg ref={svgRef}></svg>
    </div>
  );
};
```

### Use in Your App
```typescript
// App.tsx
import { KnowledgeGraphVisualization } from './KnowledgeGraphVisualization';

function App() {
  return (
    <div className="App">
      <h1>AI Assistant Learning Dashboard</h1>
      <KnowledgeGraphVisualization />
    </div>
  );
}
```

---

## 4. Alternative: Vue.js Integration

```vue
<!-- KnowledgeGraph.vue -->
<template>
  <div class="knowledge-graph">
    <h2>Knowledge Graph</h2>
    <div v-if="loading">Loading...</div>
    <div v-else>
      <div class="stats">
        <p>Nodes: {{ stats.node_count }}</p>
        <p>Edges: {{ stats.edge_count }}</p>
      </div>
      <svg ref="svg" width="1200" height="800"></svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import * as d3 from 'd3';

const loading = ref(true);
const stats = ref({ node_count: 0, edge_count: 0 });
const svg = ref<SVGElement | null>(null);

onMounted(async () => {
  const response = await fetch('http://localhost:8000/api/learning/knowledge-graph/export');
  const data = await response.json();
  
  if (data.success) {
    stats.value = data.graph.stats;
    renderGraph(data.graph);
    loading.value = false;
  }
});

function renderGraph(graphData: any) {
  // Similar D3 rendering logic as React example
  // ... (see React example above)
}
</script>
```

---

## 5. Python CLI Example

```python
#!/usr/bin/env python3
"""
Knowledge Graph CLI Viewer
Quick command-line tool to inspect the knowledge graph
"""

import requests
import json
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

console = Console()

def view_graph_stats():
    """Display knowledge graph statistics"""
    response = requests.get('http://localhost:8000/api/learning/knowledge-graph/stats')
    data = response.json()
    
    if data['success']:
        stats = data['stats']
        
        table = Table(title="Knowledge Graph Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key, str(value))
        
        console.print(table)

def view_graph_data():
    """Display full graph data"""
    response = requests.get('http://localhost:8000/api/learning/knowledge-graph/export')
    data = response.json()
    
    if data['success']:
        graph = data['graph']
        
        # Display nodes
        nodes_table = Table(title="Knowledge Graph Nodes (Top 10)")
        nodes_table.add_column("ID", style="cyan")
        nodes_table.add_column("Type", style="yellow")
        nodes_table.add_column("Content", style="green")
        nodes_table.add_column("Importance", style="magenta")
        
        for node in graph['nodes'][:10]:
            nodes_table.add_row(
                node['id'][:20],
                node.get('type', 'unknown'),
                node.get('content', '')[:40],
                f"{node.get('importance_score', 0):.2f}"
            )
        
        console.print(nodes_table)
        
        # Display edges
        edges_table = Table(title="Knowledge Graph Edges (Top 10)")
        edges_table.add_column("Source", style="cyan")
        edges_table.add_column("Target", style="yellow")
        edges_table.add_column("Relationship", style="green")
        edges_table.add_column("Strength", style="magenta")
        
        for link in graph['links'][:10]:
            edges_table.add_row(
                str(link['source'])[:20],
                str(link['target'])[:20],
                link.get('relationship_type', 'unknown'),
                f"{link.get('strength', 0):.2f}"
            )
        
        console.print(edges_table)

if __name__ == '__main__':
    console.print("[bold blue]Knowledge Graph Viewer[/bold blue]\n")
    view_graph_stats()
    console.print()
    view_graph_data()
```

Run it:
```bash
pip install rich requests
python view_knowledge_graph.py
```

---

## 6. Testing Checklist

### Backend Tests
```bash
# Test 1: Feedback system initialized
python -c "from ai_assistant.modules.conversational_ai import AdvancedConversationalAI; ai = AdvancedConversationalAI(); print(f'Feedback: {ai.feedback_system is not None}')"

# Test 2: API endpoints respond
curl http://localhost:8000/api/learning/knowledge-graph/export | jq .success

# Test 3: Dashboard works
curl http://localhost:8000/api/learning/dashboard | jq .timestamp
```

### Integration Tests
```python
# test_integration.py
import requests

def test_knowledge_graph_export():
    response = requests.get('http://localhost:8000/api/learning/knowledge-graph/export')
    assert response.status_code == 200
    data = response.json()
    assert data['success'] == True
    assert 'nodes' in data['graph']
    assert 'links' in data['graph']

def test_feedback_logging():
    from ai_assistant.modules.conversational_ai import AdvancedConversationalAI
    ai = AdvancedConversationalAI()
    
    # Process message
    response = ai.process_message("test message")
    
    # Check feedback system logged it
    assert ai.feedback_system is not None
    stats = ai.feedback_system.get_stats()
    assert stats['total_interactions'] > 0

if __name__ == '__main__':
    test_knowledge_graph_export()
    test_feedback_logging()
    print("✅ All integration tests passed!")
```

---

## 7. Production Deployment

### Docker Setup
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "ai_assistant.services.learning_api:router", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./ai_assistant/databases:/app/ai_assistant/databases
    environment:
      - PYTHONUNBUFFERED=1
  
  frontend:
    image: node:18
    working_dir: /app
    volumes:
      - ./project:/app
    command: npm run dev
    ports:
      - "5173:5173"
```

Run:
```bash
docker-compose up -d
```

---

## 8. Monitoring & Analytics

### Add Logging
```python
# In conversational_ai.py, enhance feedback logging:
import logging

logger = logging.getLogger(__name__)

def process_message(self, message: str, role: str = "user") -> str:
    # ... existing code ...
    
    if self.feedback_system and role == "user":
        try:
            self.feedback_system.log_interaction(message, response, context)
            logger.info(f"Logged interaction: {message[:50]}...")
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
```

### Analytics Dashboard API
Add to [learning_api.py](ai_assistant/services/learning_api.py):

```python
@router.get("/analytics/interactions")
async def get_interaction_analytics(days: int = 7):
    """Get interaction analytics for the last N days"""
    from datetime import datetime, timedelta
    
    engine = AdaptiveLearningEngine()
    cutoff = datetime.now() - timedelta(days=days)
    
    interactions = engine.get_interactions_since(cutoff)
    
    return {
        "success": True,
        "period": f"Last {days} days",
        "total_interactions": len(interactions),
        "by_type": {
            "command": sum(1 for i in interactions if i.get('context', {}).get('type') == 'command'),
            "conversation": sum(1 for i in interactions if i.get('context', {}).get('type') == 'conversation'),
            "math": sum(1 for i in interactions if i.get('context', {}).get('type') == 'math'),
        }
    }
```

---

## Summary

**Backend**: ✅ Complete - Both features fully integrated  
**API**: ✅ Complete - REST endpoints operational  
**Frontend**: ⏳ Ready for integration (examples provided)  
**Testing**: ⏳ Integration tests ready  
**Deployment**: ⏳ Docker configuration ready  

Next step: Choose your frontend framework and implement the visualization dashboard!
