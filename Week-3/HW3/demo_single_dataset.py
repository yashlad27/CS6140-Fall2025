#!/usr/bin/env python3
"""
Simple demo script to run Kernel PCA on a single dataset
"""

from kernel_pca_basics import KernelPCAVisualizer

def demo_circles():
    """
    Demonstrate Kernel PCA on circles dataset - shows RBF kernel superiority
    """
    print("🔵 CIRCLES DATASET DEMO - Perfect for RBF Kernel")
    print("=" * 50)
    
    visualizer = KernelPCAVisualizer()
    
    # Create concentric circles data
    X, y = visualizer.create_synthetic_data('circles', n_samples=200, noise=0.05)
    
    print("\n📊 Step 1: Visualizing Original Data")
    visualizer.visualize_original_data()
    
    print("\n🔄 Step 2: Understanding RBF Kernel Transformation")
    visualizer.apply_kernel_transformation('rbf', gamma=1.0)
    
    print("\n📈 Step 3: Comparing All PCA Methods")
    results = visualizer.compare_pca_methods()
    
    print("\n💡 Step 4: Understanding Why Kernel PCA Works Better")
    visualizer.demonstrate_kernel_benefits()
    
    print("\n✅ Demo completed! Key observations:")
    print("   • Standard PCA fails to separate circular patterns")
    print("   • RBF Kernel PCA perfectly separates the circles")
    print("   • Higher γ values create more localized transformations")

if __name__ == "__main__":
    demo_circles()
