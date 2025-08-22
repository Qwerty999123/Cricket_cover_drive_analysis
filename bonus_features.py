import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import deque
from scipy.signal import find_peaks, savgol_filter

from basic_analyzer import BiomechanicalMetrics, ShotEvaluation, CricketAnalyzer

@dataclass
class PhaseInfo:
    """Information about a detected cricket shot phase"""
    phase_name: str
    start_frame: int
    end_frame: int
    duration: float
    confidence: float
    key_metrics: Dict[str, float]

class PhaseDetector:
    """Detects cricket shot phases automatically"""
    
    def __init__(self):
        self.velocity_window = 5
        self.angle_threshold = 10  # degrees per frame
        self.velocity_threshold = 20  # pixels per frame
        
        # Phase definitions
        self.phases = [
            "stance", "stride", "downswing", "impact", "follow_through", "recovery"
        ]
    
    def calculate_joint_velocities(self, metrics_history: List[BiomechanicalMetrics]) -> Dict[str, List[float]]:
        """Calculate joint velocities from position history"""
        velocities = {
            'elbow_angle_velocity': [],
            'spine_lean_velocity': [],
            'head_velocity': []
        }
        
        for i in range(1, len(metrics_history)):
            current = metrics_history[i]
            previous = metrics_history[i-1]
            
            dt = current.timestamp - previous.timestamp
            if dt > 0:
                # Angular velocities
                if current.front_elbow_angle and previous.front_elbow_angle:
                    elbow_vel = abs(current.front_elbow_angle - previous.front_elbow_angle) / dt
                    velocities['elbow_angle_velocity'].append(elbow_vel)
                else:
                    velocities['elbow_angle_velocity'].append(0)
                
                if current.spine_lean and previous.spine_lean:
                    spine_vel = abs(current.spine_lean - previous.spine_lean) / dt
                    velocities['spine_lean_velocity'].append(spine_vel)
                else:
                    velocities['spine_lean_velocity'].append(0)
                
                # Head movement velocity (simplified)
                if current.head_knee_alignment and previous.head_knee_alignment:
                    head_vel = abs(current.head_knee_alignment - previous.head_knee_alignment) / dt
                    velocities['head_velocity'].append(head_vel)
                else:
                    velocities['head_velocity'].append(0)
        
        return velocities
    
    def detect_phases(self, metrics_history: List[BiomechanicalMetrics]) -> List[PhaseInfo]:
        """Detect cricket shot phases using velocity and angle analysis"""
        if len(metrics_history) < 10:
            return []
        
        phases = []
        velocities = self.calculate_joint_velocities(metrics_history)
        
        # Smooth the velocity signals
        elbow_velocity = np.array(velocities['elbow_angle_velocity'])
        if len(elbow_velocity) > 5:
            elbow_velocity = savgol_filter(elbow_velocity, 5, 2)
        
        # Find velocity peaks for phase transitions
        peaks, _ = find_peaks(elbow_velocity, height=self.velocity_threshold, distance=10)
        
        # Define phases based on peaks and patterns
        if len(peaks) > 0:
            # Stance phase (start to first major movement)
            stance_end = peaks[0] if peaks[0] > 5 else 10
            phases.append(PhaseInfo(
                phase_name="stance",
                start_frame=0,
                end_frame=stance_end,
                duration=metrics_history[stance_end].timestamp - metrics_history[0].timestamp,
                confidence=0.8,
                key_metrics={"avg_spine_lean": np.mean([m.spine_lean for m in metrics_history[:stance_end] if m.spine_lean])}
            ))
            
            # Find impact phase (highest velocity)
            if len(peaks) > 1:
                impact_frame = peaks[np.argmax(elbow_velocity[peaks])]
                
                # Downswing (before impact)
                downswing_start = max(stance_end, impact_frame - 15)
                phases.append(PhaseInfo(
                    phase_name="downswing",
                    start_frame=downswing_start,
                    end_frame=impact_frame,
                    duration=metrics_history[impact_frame].timestamp - metrics_history[downswing_start].timestamp,
                    confidence=0.9,
                    key_metrics={"max_elbow_velocity": float(np.max(elbow_velocity[downswing_start:impact_frame]))}
                ))
                
                # Impact phase
                impact_end = min(impact_frame + 5, len(metrics_history) - 1)
                phases.append(PhaseInfo(
                    phase_name="impact",
                    start_frame=impact_frame,
                    end_frame=impact_end,
                    duration=metrics_history[impact_end].timestamp - metrics_history[impact_frame].timestamp,
                    confidence=0.95,
                    key_metrics={"impact_velocity": float(elbow_velocity[impact_frame])}
                ))
                
                # Follow-through
                if impact_end < len(metrics_history) - 5:
                    follow_end = min(impact_end + 20, len(metrics_history) - 1)
                    phases.append(PhaseInfo(
                        phase_name="follow_through",
                        start_frame=impact_end,
                        end_frame=follow_end,
                        duration=metrics_history[follow_end].timestamp - metrics_history[impact_end].timestamp,
                        confidence=0.8,
                        key_metrics={"follow_through_angle_change": float(np.std(elbow_velocity[impact_end:follow_end]))}
                    ))
        
        return phases

class TemporalAnalyzer:
    """Analyzes temporal consistency and smoothness"""
    
    def __init__(self):
        self.smoothness_window = 10
    
    def calculate_smoothness_metrics(self, metrics_history: List[BiomechanicalMetrics]) -> Dict[str, float]:
        """Calculate smoothness metrics for the shot"""
        if len(metrics_history) < self.smoothness_window:
            return {}
        
        smoothness = {}
        
        # Extract angle sequences
        elbow_angles = [m.front_elbow_angle for m in metrics_history if m.front_elbow_angle is not None]
        spine_leans = [m.spine_lean for m in metrics_history if m.spine_lean is not None]
        
        if len(elbow_angles) > 5:
            # Calculate frame-to-frame angle differences
            elbow_diffs = np.diff(elbow_angles)
            smoothness['elbow_angle_smoothness'] = 1.0 / (1.0 + np.std(elbow_diffs))
            smoothness['elbow_angle_variance'] = float(np.var(elbow_angles))
            
        if len(spine_leans) > 5:
            spine_diffs = np.diff(spine_leans)
            smoothness['spine_lean_smoothness'] = 1.0 / (1.0 + np.std(spine_diffs))
            smoothness['spine_lean_variance'] = float(np.var(spine_leans))
        
        # Overall smoothness score
        if smoothness:
            smoothness['overall_smoothness'] = np.mean(list(smoothness.values()))
        
        return smoothness
    
    def generate_temporal_chart(self, metrics_history: List[BiomechanicalMetrics], output_path: str):
        """Generate temporal analysis charts"""
        if len(metrics_history) < 5:
            return
        
        # Extract data
        timestamps = [m.timestamp for m in metrics_history]
        elbow_angles = [m.front_elbow_angle if m.front_elbow_angle else np.nan for m in metrics_history]
        spine_leans = [m.spine_lean if m.spine_lean else np.nan for m in metrics_history]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Elbow angle plot
        ax1.plot(timestamps, elbow_angles, 'b-', linewidth=2, label='Front Elbow Angle')
        ax1.fill_between(timestamps, 100, 140, alpha=0.2, color='green', label='Ideal Range')
        ax1.set_ylabel('Angle (degrees)')
        ax1.set_title('Front Elbow Angle Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Spine lean plot
        ax2.plot(timestamps, spine_leans, 'r-', linewidth=2, label='Spine Lean')
        ax2.fill_between(timestamps, 0, 15, alpha=0.2, color='green', label='Good Range')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Lean (degrees)')
        ax2.set_title('Spine Lean Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

class SkillGrader:
    """Maps performance metrics to skill levels"""
    
    def __init__(self):
        self.skill_thresholds = {
            'advanced': 8.0,
            'intermediate': 6.0,
            'beginner': 0.0
        }
    
    def grade_skill_level(self, evaluation: ShotEvaluation) -> Dict[str, Any]:
        """Determine skill level based on evaluation"""
        overall_score = evaluation.overall_score
        
        # Determine overall skill level
        if overall_score >= self.skill_thresholds['advanced']:
            skill_level = 'advanced'
            description = "Excellent technique with strong fundamentals"
        elif overall_score >= self.skill_thresholds['intermediate']:
            skill_level = 'intermediate'
            description = "Good technique with room for improvement"
        else:
            skill_level = 'beginner'
            description = "Developing technique, focus on basics"
        
        # Detailed breakdown
        category_grades = {}
        for category in ['footwork', 'head_position', 'swing_control', 'balance', 'follow_through']:
            score = getattr(evaluation, f'{category}_score')
            if score >= 8.0:
                category_grades[category] = 'advanced'
            elif score >= 6.0:
                category_grades[category] = 'intermediate'
            else:
                category_grades[category] = 'beginner'
        
        # Identify strengths and weaknesses
        strengths = [cat for cat, grade in category_grades.items() if grade == 'advanced']
        weaknesses = [cat for cat, grade in category_grades.items() if grade == 'beginner']
        
        return {
            'overall_skill_level': skill_level,
            'overall_score': overall_score,
            'description': description,
            'category_grades': category_grades,
            'strengths': strengths,
            'areas_for_improvement': weaknesses,
            'recommendations': self._generate_recommendations(skill_level, weaknesses)
        }
    
    def _generate_recommendations(self, skill_level: str, weaknesses: List[str]) -> List[str]:
        """Generate skill-specific recommendations"""
        recommendations = []
        
        base_recommendations = {
            'footwork': "Practice getting to the pitch of the ball with decisive footwork",
            'head_position': "Focus on keeping your head steady and over the front knee",
            'swing_control': "Work on maintaining high front elbow throughout the shot",
            'balance': "Practice staying balanced throughout the stroke",
            'follow_through': "Ensure complete follow-through toward your target"
        }
        
        skill_specific = {
            'beginner': {
                'footwork': "Start with stationary ball drills to build basic footwork patterns",
                'head_position': "Practice shadow batting focusing on head position",
                'swing_control': "Use a shorter bat or tennis ball for control practice",
                'balance': "Work on basic stance and weight transfer drills",
                'follow_through': "Practice slow-motion shots focusing on complete follow-through"
            },
            'intermediate': {
                'footwork': "Practice moving to different line and length combinations",
                'head_position': "Work on maintaining head position under pressure",
                'swing_control': "Focus on timing and bat speed control",
                'balance': "Practice shots on uneven surfaces for better balance",
                'follow_through': "Work on varying follow-through for different shot directions"
            },
            'advanced': {
                'footwork': "Practice advanced footwork patterns for different game situations",
                'head_position': "Focus on head position consistency in match scenarios",
                'swing_control': "Work on shot variations and advanced timing",
                'balance': "Practice maintaining balance in dynamic game situations",
                'follow_through': "Refine follow-through for maximum power and control"
            }
        }
        
        # Add specific recommendations for weaknesses
        for weakness in weaknesses:
            if weakness in skill_specific.get(skill_level, {}):
                recommendations.append(skill_specific[skill_level][weakness])
            else:
                recommendations.append(base_recommendations.get(weakness, f"Work on improving {weakness}"))
        
        # Add general recommendation based on skill level
        if skill_level == 'beginner':
            recommendations.append("Focus on mastering basic technique before advancing to complex drills")
        elif skill_level == 'intermediate':
            recommendations.append("Practice consistently and work on weak areas while maintaining strengths")
        else:
            recommendations.append("Fine-tune technique and focus on game application")
        
        return recommendations

class ReportGenerator:
    """Generates comprehensive analysis reports"""
    
    def __init__(self):
        self.template_dir = "templates"
        os.makedirs(self.template_dir, exist_ok=True)
    
    def generate_html_report(self, 
                           evaluation: ShotEvaluation,
                           skill_grade: Dict[str, Any],
                           phases: List[PhaseInfo],
                           smoothness: Dict[str, float],
                           output_path: str):
        """Generate comprehensive HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Cricket Cover Drive Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .score-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .score-large {{ font-size: 48px; font-weight: bold; color: #007bff; }}
                .category-scores {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .category {{ background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
                .skill-level {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .recommendations {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .phase-timeline {{ margin: 20px 0; }}
                .phase {{ background: #f1f3f4; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .contact-moment {{ background: #ffe6e6; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .good {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .poor {{ color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                
                <div class="score-card">
                    <h2>Overall Performance</h2>
                    <div style="text-align: center;">
                        <div class="score-large">{evaluation.overall_score:.1f}/10</div>
                        <p><strong>Skill Level: {skill_grade['overall_skill_level'].title()}</strong></p>
                        <p>{skill_grade['description']}</p>
                    </div>
                </div>
                
                <div class="category-scores">
                    <div class="category">
                        <h3>Footwork</h3>
                        <div class="score-large" style="font-size: 24px;">{evaluation.footwork_score:.1f}/10</div>
                        <p>{evaluation.footwork_feedback}</p>
                    </div>
                    <div class="category">
                        <h3>Head Position</h3>
                        <div class="score-large" style="font-size: 24px;">{evaluation.head_position_score:.1f}/10</div>
                        <p>{evaluation.head_position_feedback}</p>
                    </div>
                    <div class="category">
                        <h3>Swing Control</h3>
                        <div class="score-large" style="font-size: 24px;">{evaluation.swing_control_score:.1f}/10</div>
                        <p>{evaluation.swing_control_feedback}</p>
                    </div>
                    <div class="category">
                        <h3>Balance</h3>
                        <div class="score-large" style="font-size: 24px;">{evaluation.balance_score:.1f}/10</div>
                        <p>{evaluation.balance_feedback}</p>
                    </div>
                    <div class="category">
                        <h3>Follow Through</h3>
                        <div class="score-large" style="font-size: 24px;">{evaluation.follow_through_score:.1f}/10</div>
                        <p>{evaluation.follow_through_feedback}</p>
                    </div>
                </div>
        """
        
        # Add skill level details
        html_content += f"""
                <div class="skill-level">
                    <h2>Skill Assessment</h2>
                    <table>
                        <tr><th>Category</th><th>Grade</th></tr>
        """
        
        for category, grade in skill_grade['category_grades'].items():
            html_content += f"<tr><td>{category.replace('_', ' ').title()}</td><td>{grade.title()}</td></tr>"
        
        html_content += "</table>"
        
        if skill_grade['strengths']:
            html_content += f"<p><strong>Strengths:</strong> {', '.join([s.replace('_', ' ').title() for s in skill_grade['strengths']])}</p>"
        
        if skill_grade['areas_for_improvement']:
            html_content += f"<p><strong>Areas for Improvement:</strong> {', '.join([w.replace('_', ' ').title() for w in skill_grade['areas_for_improvement']])}</p>"
        
        html_content += "</div>"
        
        # Add recommendations
        html_content += """
                <div class="recommendations">
                    <h2>Training Recommendations</h2>
                    <ul>
        """
        
        for rec in skill_grade['recommendations']:
            html_content += f"<li>{rec}</li>"
        
        html_content += "</ul></div>"
        
        # Add phase analysis if available
        if phases:
            html_content += """
                    <div class="phase-timeline">
                        <h2>Shot Phase Analysis</h2>
            """
            
            for phase in phases:
                html_content += f"""
                        <div class="phase">
                            <strong>{phase.phase_name.title()}</strong> 
                            (Frames {phase.start_frame}-{phase.end_frame}, {phase.duration:.2f}s)
                            - Confidence: {phase.confidence:.2f}
                        </div>
                """
            
            html_content += "</div>"
        
        # Add smoothness metrics if available
        if smoothness:
            html_content += """
                    <div>
                        <h2>Temporal Analysis</h2>
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
            """
            
            for metric, value in smoothness.items():
                html_content += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.3f}</td></tr>"
            
            html_content += "</table></div>"
        
        
        with open(output_path, 'w') as f:
            f.write(html_content)

class AdvancedCricketAnalyzer:
    """Enhanced analyzer with bonus features"""
    
    def __init__(self, output_dir: str = "output"):
        # Import the base analyzer
        
        self.base_analyzer = CricketAnalyzer(output_dir)
        self.output_dir = output_dir
        
        # Initialize bonus components
        self.phase_detector = PhaseDetector()
        self.temporal_analyzer = TemporalAnalyzer()
        self.skill_grader = SkillGrader()
        self.report_generator = ReportGenerator()
        
        # Enable bonus features
        self.enable_phase_detection = True
        self.enable_contact_detection = True
        self.enable_temporal_analysis = True
        self.enable_skill_grading = True
        self.generate_html_report = True
    
    def analyze_video_advanced(self, video_path: str) -> Dict[str, Any]:
        """Run enhanced analysis with bonus features"""
        
        # Run base analysis
        base_results = self.base_analyzer.analyze_video(video_path)
        
        # Get metrics history
        metrics_history = self.base_analyzer.shot_evaluator.metrics_history
        evaluation = self.base_analyzer.shot_evaluator.evaluate_shot()
        
        # Enhanced analysis
        enhanced_results = base_results.copy()
        
        # Phase detection
        if self.enable_phase_detection and len(metrics_history) > 10:
            phases = self.phase_detector.detect_phases(metrics_history)
            enhanced_results['phases'] = [
                {
                    'phase_name': p.phase_name,
                    'start_frame': p.start_frame,
                    'end_frame': p.end_frame,
                    'duration': p.duration,
                    'confidence': p.confidence,
                    'key_metrics': p.key_metrics
                } for p in phases
            ]
        else:
            phases = []
        
        # Temporal analysis
        if self.enable_temporal_analysis and len(metrics_history) > 10:
            smoothness = self.temporal_analyzer.calculate_smoothness_metrics(metrics_history)
            enhanced_results['smoothness_metrics'] = smoothness
            
            # Generate temporal chart
            chart_path = os.path.join(self.output_dir, "temporal_analysis.png")
            self.temporal_analyzer.generate_temporal_chart(metrics_history, chart_path)
            enhanced_results['temporal_chart_path'] = chart_path
        else:
            smoothness = {}
        
        # Skill grading
        if self.enable_skill_grading:
            skill_grade = self.skill_grader.grade_skill_level(evaluation)
            enhanced_results['skill_assessment'] = skill_grade
        else:
            skill_grade = {}
        
        # Generate HTML report
        if self.generate_html_report:
            report_path = os.path.join(self.output_dir, "analysis_report.html")
            self.report_generator.generate_html_report(
                evaluation, skill_grade, phases, smoothness, report_path
            )
            enhanced_results['html_report_path'] = report_path
        
        return enhanced_results