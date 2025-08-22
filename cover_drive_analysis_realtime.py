import os
import sys
import argparse
import time

from basic_analyzer import CricketAnalyzer
from video_utility import download_target_video
from bonus_features import AdvancedCricketAnalyzer

def print_banner():
    """Print application banner"""
    print("=" * 58)
    print("    AthleteRise - AI-Powered Cricket Analytics")
    print("         Real-Time Cover Drive Analysis")
    print("=" * 58)
    print()

def run_basic_analysis(video_path, output_dir, verbose=False):
    """Run basic analysis"""
    print("\nRunning basic analysis...")
    
    analyzer = CricketAnalyzer(output_dir=output_dir)
    
    start_time = time.time()
    results = analyzer.analyze_video(video_path)
    analysis_time = time.time() - start_time
    
    print(f"Basic analysis complete in {analysis_time:.1f}s")
    print(f"Overall Score: {results['evaluation']['overall_score']:.1f}/10")
    print(f"Processing FPS: {results['performance']['avg_fps']:.1f}")
    
    return results

def run_advanced_analysis(video_path, output_dir, verbose=False):
    """Run advanced analysis with bonus features"""
    print("\nRunning advanced analysis...")
    
    analyzer = AdvancedCricketAnalyzer(output_dir=output_dir)
    
    start_time = time.time()
    results = analyzer.analyze_video_advanced(video_path)
    analysis_time = time.time() - start_time
    
    print(f"Advanced analysis complete in {analysis_time:.1f}s")
    
    # Print comprehensive results
    evaluation = results['evaluation']
    print(f"Overall Score: {evaluation['overall_score']:.1f}/10")
    
    if 'skill_assessment' in results:
        skill = results['skill_assessment']
        print(f"Skill Level: {skill['overall_skill_level'].title()}")
        
        if skill['strengths']:
            print(f"Strengths: {', '.join([s.replace('_', ' ').title() for s in skill['strengths']])}")
        
        if skill['areas_for_improvement']:
            print(f"Areas for Improvement: {', '.join([w.replace('_', ' ').title() for w in skill['areas_for_improvement']])}")
    
    if 'phases' in results:
        print(f"Shot Phases Detected: {len(results['phases'])}")
        for phase in results['phases']:
            print(f"   - {phase['phase_name'].title()}: {phase['duration']:.2f}s (confidence: {phase['confidence']:.2f})")
    
    if 'smoothness_metrics' in results:
        smoothness = results['smoothness_metrics']
        if 'overall_smoothness' in smoothness:
            print(f"Technique Smoothness: {smoothness['overall_smoothness']:.3f}")
    
    return results

def print_results_summary(results, output_dir):
    """Print final results summary"""
    print("\n" + "=" * 58)
    print("                    ANALYSIS COMPLETE")
    print("=" * 58)
    
    evaluation = results['evaluation']
    
    # Overall score with visual indicator
    score = evaluation['overall_score']
    if score >= 8.0:
        score_desc = "Excellent"
    elif score >= 6.0:
        score_desc = "Good"
    elif score >= 4.0:
        score_desc = "Needs Work"
    else:
        score_desc = "Poor"
    
    print(f"\nOVERALL SCORE: {score:.1f}/10 ({score_desc})")
    
    # Category breakdown
    categories = [
        ('Footwork', evaluation['footwork_score']),
        ('Head Position', evaluation['head_position_score']),
        ('Swing Control', evaluation['swing_control_score']),
        ('Balance', evaluation['balance_score']),
        ('Follow Through', evaluation['follow_through_score'])
    ]
    
    print(f"\nCATEGORY BREAKDOWN:")
    for name, score in categories:
        print(f"   {name:15}:{score:.1f}/10")
    
    # Output files
    print(f"\nOUTPUT FILES:")
    print(f"   Annotated Video: {results.get('video_path', 'N/A')}")
    print(f"   Evaluation JSON: {results.get('evaluation_path', 'N/A')}")
    
    if 'html_report_path' in results:
        print(f"   HTML Report: {results['html_report_path']}")
    
    if 'temporal_chart_path' in results:
        print(f"   Temporal Chart: {results['temporal_chart_path']}")
    
    # Performance stats
    if 'performance' in results:
        perf = results['performance']
        print(f"\nPERFORMANCE:")
        print(f"   Processing FPS: {perf.get('avg_fps', 0):.1f}")
        print(f"   Total Frames: {perf.get('total_frames', 0)}")
        print(f"   Analysis Time: {perf.get('processing_time', 0):.1f}s")
        
        if perf.get('target_fps_achieved', False):
            print(f"   Real-time target achieved!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AthleteRise Cricket Analytics - Complete Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input video.mp4                    # Basic analysis
  %(prog)s --input video.mp4 --advanced         # Advanced analysis  
  %(prog)s --download --advanced                # Download target video and analyze
  %(prog)s --streamlit                          # Launch web interface
  %(prog)s --setup                              # Run system setup
        """
    )
    
    parser.add_argument("--input", "-i", help="Input video file path")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--download", "-d", action="store_true", 
                       help="Download target video from YouTube")
    parser.add_argument("--advanced", "-a", action="store_true",
                       help="Use advanced analysis with bonus features")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--url", help="YouTube URL to download")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Determine video path
    video_path = None

    if args.download:
        video_path = download_target_video(args.output)
        if not video_path:
            print("Failed to download video")
            sys.exit(1)
    elif args.url:
        video_path = download_target_video(output_dir= args.output, url=args.url)
        if not video_path:
            print("Failed to download video")
            sys.exit(1)
    elif args.input:
        video_path = args.input
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            sys.exit(1)
    else:
        # Try to find video automatically
        video_path = download_target_video(args.output)
        if not video_path:
            print("No video specified and none found.")
            print("Use --input <video_file> or --download")
            sys.exit(1)
    
    print(f"Analyzing video: {video_path}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run analysis
    try:
        if args.advanced:
            results = run_advanced_analysis(video_path, args.output, args.verbose)
        else:
            results = run_basic_analysis(video_path, args.output, args.verbose)
        
        # Print results
        print_results_summary(results, args.output)
        
        # Success message
        print(f"\nAnalysis complete")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()