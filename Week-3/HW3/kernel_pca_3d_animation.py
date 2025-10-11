#!/usr/bin/env python3
"""
3D Animation of Kernel PCA - Visualizing the transformation process
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KernelPCA3DAnimator:
    """
    Create 3D animations showing how Kernel PCA works under the hood
    """
    
    def __init__(self):
        self.data = None
        self.labels = None
        self.scaler = StandardScaler()
        
    def create_data(self, dataset_type='circles', n_samples=150):
        """Create synthetic data for visualization"""
        if dataset_type == 'circles':
            X, y = make_circles(n_samples=n_samples, noise=0.05, factor=0.3, random_state=42)
        elif dataset_type == 'moons':
            X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
        
        X = self.scaler.fit_transform(X)
        self.data = X
        self.labels = y
        return X, y
    
    def polynomial_feature_map(self, X, degree=2):
        """
        Explicit polynomial feature mapping for visualization
        Maps 2D data to higher dimensional space
        """
        x1, x2 = X[:, 0], X[:, 1]
        
        if degree == 2:
            # Map to 6D: [1, x1, x2, x1^2, x1*x2, x2^2]
            features = np.column_stack([
                np.ones(len(X)),  # bias
                x1,               # x1
                x2,               # x2
                x1**2,            # x1^2
                x1*x2,            # x1*x2
                x2**2             # x2^2
            ])
        elif degree == 3:
            # Map to 10D: add cubic terms
            features = np.column_stack([
                np.ones(len(X)),  # bias
                x1, x2,           # linear
                x1**2, x1*x2, x2**2,  # quadratic
                x1**3, x1**2*x2, x1*x2**2, x2**3  # cubic
            ])
        
        return features
    
    def rbf_feature_map_approximation(self, X, gamma=1.0, n_components=100):
        """
        Approximate RBF feature mapping using random Fourier features
        """
        n_samples, n_features = X.shape
        
        # Random weights for Fourier features
        np.random.seed(42)
        W = np.random.normal(0, np.sqrt(2 * gamma), (n_features, n_components))
        b = np.random.uniform(0, 2 * np.pi, n_components)
        
        # Compute features: sqrt(2/n_components) * cos(X @ W + b)
        features = np.sqrt(2.0 / n_components) * np.cos(X @ W + b)
        return features
    
    def animate_polynomial_transformation(self):
        """
        Animate the polynomial kernel transformation process
        """
        print("🎬 Creating polynomial kernel transformation animation...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 6))
        
        # Original 2D data
        ax1 = fig.add_subplot(141)
        ax1.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', s=50)
        ax1.set_title('Original 2D Data', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.grid(True, alpha=0.3)
        
        # 3D feature space (using first 3 polynomial features for visualization)
        ax2 = fig.add_subplot(142, projection='3d')
        poly_features = self.polynomial_feature_map(self.data, degree=2)
        
        # Use x1, x2, x1^2 for 3D visualization
        ax2.scatter(poly_features[:, 1], poly_features[:, 2], poly_features[:, 3], 
                   c=self.labels, cmap='viridis', s=50)
        ax2.set_title('3D Feature Space\n(x₁, x₂, x₁²)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        ax2.set_zlabel('x₁²')
        
        # PCA in feature space
        ax3 = fig.add_subplot(143, projection='3d')
        pca_3d = PCA(n_components=3)
        poly_pca = pca_3d.fit_transform(poly_features)
        
        ax3.scatter(poly_pca[:, 0], poly_pca[:, 1], poly_pca[:, 2], 
                   c=self.labels, cmap='viridis', s=50)
        ax3.set_title('PCA in Feature Space', fontsize=12, fontweight='bold')
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_zlabel('PC3')
        
        # Final 2D projection
        ax4 = fig.add_subplot(144)
        kpca = KernelPCA(n_components=2, kernel='poly', degree=2)
        X_kpca = kpca.fit_transform(self.data)
        
        ax4.scatter(X_kpca[:, 0], X_kpca[:, 1], c=self.labels, cmap='viridis', s=50)
        ax4.set_title('Kernel PCA Result\n(2D Projection)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Kernel PC1')
        ax4.set_ylabel('Kernel PC2')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_interactive_3d_transformation(self):
        """
        Create an interactive 3D visualization showing the transformation
        """
        print("🎬 Creating interactive 3D transformation...")
        
        # Create the transformation steps
        steps = self._get_transformation_steps()
        
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots for different views
        ax1 = fig.add_subplot(221)  # Original data
        ax2 = fig.add_subplot(222, projection='3d')  # 3D feature space
        ax3 = fig.add_subplot(223, projection='3d')  # PCA in feature space
        ax4 = fig.add_subplot(224)  # Final result
        
        self._plot_transformation_steps(ax1, ax2, ax3, ax4, steps)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _get_transformation_steps(self):
        """Get all transformation steps for visualization"""
        steps = {}
        
        # Step 1: Original data
        steps['original'] = self.data
        
        # Step 2: Polynomial features (6D -> show 3D projection)
        poly_features = self.polynomial_feature_map(self.data, degree=2)
        steps['poly_features'] = poly_features
        
        # Step 3: PCA in feature space
        pca_feature = PCA(n_components=3)
        poly_pca = pca_feature.fit_transform(poly_features)
        steps['feature_pca'] = poly_pca
        
        # Step 4: Kernel PCA result
        kpca = KernelPCA(n_components=2, kernel='poly', degree=2)
        X_kpca = kpca.fit_transform(self.data)
        steps['kernel_pca'] = X_kpca
        
        return steps
    
    def _plot_transformation_steps(self, ax1, ax2, ax3, ax4, steps):
        """Plot all transformation steps"""
        
        # Step 1: Original 2D data
        scatter1 = ax1.scatter(steps['original'][:, 0], steps['original'][:, 1], 
                              c=self.labels, cmap='viridis', s=60, alpha=0.8)
        ax1.set_title('Step 1: Original 2D Data', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.grid(True, alpha=0.3)
        
        # Step 2: 3D feature space (x1, x2, x1^2)
        poly_features = steps['poly_features']
        scatter2 = ax2.scatter(poly_features[:, 1], poly_features[:, 2], poly_features[:, 3],
                              c=self.labels, cmap='viridis', s=60, alpha=0.8)
        ax2.set_title('Step 2: Polynomial Feature Space\n(x₁, x₂, x₁²)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        ax2.set_zlabel('x₁²')
        
        # Step 3: PCA in feature space
        feature_pca = steps['feature_pca']
        scatter3 = ax3.scatter(feature_pca[:, 0], feature_pca[:, 1], feature_pca[:, 2],
                              c=self.labels, cmap='viridis', s=60, alpha=0.8)
        ax3.set_title('Step 3: PCA in Feature Space', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Feature PC1')
        ax3.set_ylabel('Feature PC2')
        ax3.set_zlabel('Feature PC3')
        
        # Step 4: Final kernel PCA result
        kernel_pca = steps['kernel_pca']
        scatter4 = ax4.scatter(kernel_pca[:, 0], kernel_pca[:, 1],
                              c=self.labels, cmap='viridis', s=60, alpha=0.8)
        ax4.set_title('Step 4: Kernel PCA Result', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Kernel PC1')
        ax4.set_ylabel('Kernel PC2')
        ax4.grid(True, alpha=0.3)
    
    def animate_rbf_transformation(self):
        """
        Animate RBF kernel transformation with rotating 3D view
        """
        print("🎬 Creating RBF kernel animation with rotation...")
        
        # Get RBF approximation
        rbf_features = self.rbf_feature_map_approximation(self.data, gamma=1.0, n_components=100)
        
        # Use PCA to get 3D visualization of high-dimensional RBF features
        pca_3d = PCA(n_components=3)
        rbf_3d = pca_3d.fit_transform(rbf_features)
        
        # Create figure
        fig = plt.figure(figsize=(16, 6))
        
        # Original data
        ax1 = fig.add_subplot(131)
        ax1.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', s=50)
        ax1.set_title('Original 2D Data', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.grid(True, alpha=0.3)
        
        # RBF feature space (3D projection)
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(rbf_3d[:, 0], rbf_3d[:, 1], rbf_3d[:, 2], 
                   c=self.labels, cmap='viridis', s=50)
        ax2.set_title('RBF Feature Space\n(3D Projection)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('RBF PC1')
        ax2.set_ylabel('RBF PC2')
        ax2.set_zlabel('RBF PC3')
        
        # Kernel PCA result
        ax3 = fig.add_subplot(133)
        kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=1.0)
        X_kpca_rbf = kpca_rbf.fit_transform(self.data)
        
        ax3.scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=self.labels, cmap='viridis', s=50)
        ax3.set_title('RBF Kernel PCA Result', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Kernel PC1')
        ax3.set_ylabel('Kernel PC2')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_step_by_step_explanation(self):
        """
        Create a comprehensive step-by-step explanation with visualizations
        """
        print("\n" + "="*60)
        print("🎓 KERNEL PCA: STEP-BY-STEP EXPLANATION")
        print("="*60)
        
        print("\n📚 What happens in Kernel PCA:")
        print("1. 📊 Start with 2D data that's not linearly separable")
        print("2. 🔄 Apply kernel function to map to higher-dimensional space")
        print("3. 📈 Perform PCA in the high-dimensional feature space")
        print("4. 📉 Project back to desired number of dimensions")
        print("5. ✨ Result: Non-linear patterns become linearly separable!")
        
        # Show the mathematical transformation
        self._show_mathematical_transformation()
        
        # Create the visualizations
        print("\n🎬 Creating visualizations...")
        
        # Polynomial kernel visualization
        print("\n1️⃣ Polynomial Kernel Transformation:")
        self.animate_polynomial_transformation()
        
        # RBF kernel visualization  
        print("\n2️⃣ RBF Kernel Transformation:")
        self.animate_rbf_transformation()
        
        # Interactive 3D
        print("\n3️⃣ Interactive 3D Transformation:")
        self.create_interactive_3d_transformation()
    
    def _show_mathematical_transformation(self):
        """Show the mathematical steps"""
        print("\n🔢 Mathematical Steps:")
        print("   Original data: X ∈ ℝ²")
        print("   Kernel function: K(x,y) = φ(x)ᵀφ(y)")
        print("   ")
        print("   For Polynomial kernel (degree=2):")
        print("   φ(x₁,x₂) = [1, x₁, x₂, x₁², x₁x₂, x₂²]ᵀ ∈ ℝ⁶")
        print("   ")
        print("   For RBF kernel:")
        print("   φ(x) maps to infinite-dimensional space")
        print("   K(x,y) = exp(-γ||x-y||²)")
        print("   ")
        print("   PCA in feature space finds principal components of φ(X)")

def demo_kernel_pca_3d():
    """
    Main demo function for 3D Kernel PCA animation
    """
    print("🚀 Starting 3D Kernel PCA Animation Demo")
    
    animator = KernelPCA3DAnimator()
    
    # Create circles dataset (perfect for demonstrating kernel benefits)
    print("\n📊 Creating concentric circles dataset...")
    X, y = animator.create_data('circles', n_samples=200)
    
    # Run the complete step-by-step explanation
    animator.create_step_by_step_explanation()
    
    print("\n✅ 3D Animation Demo Complete!")
    print("\n💡 Key Insights:")
    print("   • Kernel PCA transforms non-linearly separable data")
    print("   • Higher-dimensional feature space enables linear separation")
    print("   • Different kernels create different feature mappings")
    print("   • RBF kernel works well for circular/radial patterns")
    print("   • Polynomial kernel captures polynomial relationships")

if __name__ == "__main__":
    demo_kernel_pca_3d()
