#!/usr/bin/env python3
"""
Personal Knowledge Graph Demo
Demonstrates the capabilities of the enhanced learning system with PersonalKnowledgeGraph
"""

from automation_tools_new import PersonalKnowledgeGraph, EnhancedLearningSystem
import os

def demo_knowledge_graph():
    """Demonstrate PersonalKnowledgeGraph functionality"""
    print("🧠 Personal Knowledge Graph Demo")
    print("=" * 50)
    
    # Create knowledge graph instance via Enhanced Learning System
    learning_path = "data/enhanced_learning.db"
    os.makedirs("data", exist_ok=True)
    
    try:
        # Initialize the full Enhanced Learning System which creates the database
        els = EnhancedLearningSystem(learning_path)
        kg = els.knowledge_graph  # Get the knowledge graph component
        print(f"✅ Knowledge graph initialized: {learning_path}")
        
        # Add some sample knowledge using knowledge graph methods
        print("\n📚 Adding sample knowledge...")
        
        # Add entities to knowledge graph
        python_id = kg.add_knowledge_node("Python programming language", "concept", {
            "description": "High-level programming language",
            "use_cases": ["web_development", "data_science", "automation"],
            "difficulty": "moderate"
        })
        
        ml_id = kg.add_knowledge_node("Machine Learning", "concept", {
            "description": "AI subset focusing on learning from data",
            "prerequisites": ["statistics", "programming"],
            "applications": ["prediction", "classification", "clustering"]
        })
        
        ds_id = kg.add_knowledge_node("Data Science", "concept", {
            "description": "Interdisciplinary field using scientific methods",
            "tools": ["python", "r", "sql"],
            "domains": ["business", "research", "technology"]
        })
        
        # Add relationships
        print("\n🔗 Adding relationships...")
        kg.add_relationship(python_id, ml_id, "used_in", 0.9)
        kg.add_relationship(python_id, ds_id, "used_in", 0.8)
        kg.add_relationship(ml_id, ds_id, "part_of", 0.7)
        
        # Find related concepts
        print("\n🌐 Finding related concepts...")
        related = kg.find_related_concepts("Python", max_depth=2)
        for concept in related[:5]:  # Show first 5 related concepts
            print(f"  → {concept}")
            
        # Update knowledge from interaction
        print("\n💡 Recording learning interaction...")
        kg.update_from_interaction(
            {"topic": "python", "skill": "programming", "location": "home"},
            "code_python_script",
            "success"
        )
        
        print("\n✅ Knowledge graph demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in knowledge graph demo: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_enhanced_learning():
    """Demonstrate Enhanced Learning System"""
    print("\n\n🎯 Enhanced Learning System Demo")
    print("=" * 50)
    
    try:
        # Create learning system instance
        learning_path = "data/enhanced_learning.db"
        els = EnhancedLearningSystem(learning_path)
        print(f"✅ Enhanced Learning System initialized: {learning_path}")
        
        # Record some interactions
        print("\n📝 Recording user interactions...")
        els.learn_from_interaction(
            {"skill": "web_search", "topic": "machine learning", "time_of_day": "09:30"},
            "search_web",
            "success"
        )
        
        els.learn_from_interaction(
            {"skill": "programming", "app": "vscode", "time_of_day": "14:15"},
            "open_application", 
            "success"
        )
        
        els.learn_from_interaction(
            {"skill": "email", "recipient": "colleague", "time_of_day": "18:45"},
            "send_email",
            "failure"
        )
        
        # Get behavioral insights
        print("\n🧠 Analyzing behavioral patterns...")
        patterns = els.behavioral_learner.get_behavior_patterns()
        print(f"  - Number of learned patterns: {len(patterns)}")
        for i, pattern in enumerate(patterns[:3], 1):  # Show first 3 patterns
            print(f"  {i}. Action: {pattern.action} (confidence: {pattern.confidence:.2f})")
        
        # Get skill recommendations
        print("\n📚 Getting skill recommendations...")
        skill_recs = els.get_skill_recommendations()
        for i, skill in enumerate(skill_recs[:3], 1):
            print(f"  {i}. {skill}")
        
        # Get predictions
        print("\n🔮 Predicting next actions...")
        predictions = els.get_predictions({"time_of_day": "10:00", "day_type": "weekday"})
        for i, prediction in enumerate(predictions[:3], 1):
            action = prediction.get('action', 'Unknown')
            confidence = prediction.get('confidence', 0)
            print(f"  {i}. {action} (confidence: {confidence:.2f})")
            
        # Get knowledge insights
        print("\n💡 Knowledge insights...")
        try:
            insights = els.get_knowledge_insights()
            print(f"  - Knowledge nodes: {insights.get('total_nodes', 0)}")
            print(f"  - Knowledge connections: {insights.get('total_edges', 0)}")
        except Exception as e:
            print(f"  - Knowledge insights not available: {e}")
            
        print("\n✅ Enhanced Learning System demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in enhanced learning demo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Learning & Knowledge Graph Demo")
    print("This demo showcases the advanced AI capabilities of YourDaddy Assistant")
    print()
    
    success1 = demo_knowledge_graph()
    success2 = demo_enhanced_learning()
    
    if success1 and success2:
        print("\n🎉 All demos completed successfully!")
        print("The enhanced learning module is now fully functional.")
    else:
        print("\n⚠️ Some demos failed. Check the error messages above.")