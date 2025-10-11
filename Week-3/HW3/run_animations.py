#!/usr/bin/env python3
"""
Simple runner script for Kernel PCA 3D animations
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['numpy', 'matplotlib', 'sklearn', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install them with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def run_basic_demo():
    """Run the basic 2D Kernel PCA demo"""
    print("\n🎯 Running Basic Kernel PCA Demo...")
    try:
        from kernel_pca_basics import main
        main()
    except Exception as e:
        print(f"❌ Error running basic demo: {e}")

def run_3d_animation():
    """Run the 3D animation demo"""
    print("\n🎬 Running 3D Animation Demo...")
    try:
        from kernel_pca_3d_animation import demo_kernel_pca_3d
        demo_kernel_pca_3d()
    except Exception as e:
        print(f"❌ Error running 3D animation: {e}")

def run_rotating_demo():
    """Run the rotating 3D demo"""
    print("\n🔄 Running Rotating 3D Demo...")
    try:
        from rotating_3d_demo import main
        main()
    except Exception as e:
        print(f"❌ Error running rotating demo: {e}")

def main():
    """Main menu for running different demos"""
    print("🚀 KERNEL PCA 3D ANIMATION SUITE")
    print("="*40)
    
    if not check_dependencies():
        return
    
    while True:
        print("\n📋 Choose a demo to run:")
        print("1. 🎯 Basic Kernel PCA Comparison (2D)")
        print("2. 🎬 3D Step-by-Step Animation")
        print("3. 🔄 Rotating 3D Transformation")
        print("4. 🎪 Run All Demos")
        print("5. ❌ Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            run_basic_demo()
        elif choice == '2':
            run_3d_animation()
        elif choice == '3':
            run_rotating_demo()
        elif choice == '4':
            print("\n🎪 Running all demos...")
            run_basic_demo()
            run_3d_animation()
            run_rotating_demo()
            print("\n🎉 All demos completed!")
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
