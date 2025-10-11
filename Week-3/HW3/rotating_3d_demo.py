#!/usr/bin/env python3
"""
Rotating 3D Animation of Kernel PCA Transformation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

class RotatingKernelPCADemo:
    """
    Create rotating 3D animations of kernel transformations
    """
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def create_data(self):
        """Create concentric circles data"""
        X, y = make_circles(n_samples=150, noise=0.05, factor=0.3, random_state=42)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y
    
    def polynomial_transform_3d(self, X):
        """Transform 2D data to 3D using polynomial features"""
        x1, x2 = X[:, 0], X[:, 1]
        # Use x1, x2, x1^2 for 3D visualization
        X_3d = np.column_stack([x1, x2, x1**2])
        return X_3d
    
    def create_rotating_animation(self, save_gif=False):
        """
        Create a rotating 3D animation showing the transformation
        """
        print("🎬 Creating rotating 3D animation...")
        
        # Get data
        X, y = self.create_data()
        X_3d = self.polynomial_transform_3d(X)
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 6))
        
        # Original 2D data
        ax1 = self.fig.add_subplot(131)
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=60, alpha=0.8)
        ax1.set_title('Original 2D Data\n(Concentric Circles)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.grid(True, alpha=0.3)
        
        # 3D transformed data (this will rotate)
        self.ax = self.fig.add_subplot(132, projection='3d')
        self.scatter_3d = self.ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                                         c=y, cmap='viridis', s=60, alpha=0.8)
        self.ax.set_title('3D Feature Space\n(x₁, x₂, x₁²) - Rotating', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('x₁')
        self.ax.set_ylabel('x₂')
        self.ax.set_zlabel('x₁²')
        
        # Kernel PCA result
        ax3 = self.fig.add_subplot(133)
        kpca = KernelPCA(n_components=2, kernel='poly', degree=2)
        X_kpca = kpca.fit_transform(X)
        ax3.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', s=60, alpha=0.8)
        ax3.set_title('Kernel PCA Result\n(Linearly Separable!)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Kernel PC1')
        ax3.set_ylabel('Kernel PC2')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create animation
        def animate(frame):
            # Rotate the 3D plot
            angle = frame * 2  # 2 degrees per frame
            self.ax.view_init(elev=20, azim=angle)
            return [self.scatter_3d]
        
        # Create animation
        anim = animation.FuncAnimation(self.fig, animate, frames=180, 
                                     interval=50, blit=False, repeat=True)
        
        if save_gif:
            print("💾 Saving animation as GIF...")
            anim.save('kernel_pca_rotation.gif', writer='pillow', fps=20)
            print("✅ Saved as 'kernel_pca_rotation.gif'")
        
        plt.show()
        return anim
    
    def create_side_by_side_comparison(self):
        """
        Create side-by-side comparison of different kernel transformations
        """
        print("🎬 Creating side-by-side kernel comparison...")
        
        X, y = self.create_data()
        
        fig = plt.figure(figsize=(20, 12))
        
        # Original data
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
        ax1.set_title('Original Data', fontweight='bold')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.grid(True, alpha=0.3)
        
        # Polynomial degree 2 - 3D view
        ax2 = fig.add_subplot(2, 4, 2, projection='3d')
        X_poly2 = self.polynomial_transform_3d(X)
        ax2.scatter(X_poly2[:, 0], X_poly2[:, 1], X_poly2[:, 2], c=y, cmap='viridis', s=50)
        ax2.set_title('Polynomial Features\n(x₁, x₂, x₁²)', fontweight='bold')
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        ax2.set_zlabel('x₁²')
        
        # Polynomial degree 2 - Kernel PCA result
        ax3 = fig.add_subplot(2, 4, 3)
        kpca_poly2 = KernelPCA(n_components=2, kernel='poly', degree=2)
        X_kpca_poly2 = kpca_poly2.fit_transform(X)
        ax3.scatter(X_kpca_poly2[:, 0], X_kpca_poly2[:, 1], c=y, cmap='viridis', s=50)
        ax3.set_title('Poly Kernel PCA\n(degree=2)', fontweight='bold')
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.grid(True, alpha=0.3)
        
        # RBF kernel - approximate 3D feature space
        ax4 = fig.add_subplot(2, 4, 4, projection='3d')
        # Create RBF-like transformation for visualization
        gamma = 1.0
        centers = X[::20]  # Use subset as centers
        rbf_features = []
        for center in centers[:3]:  # Use first 3 centers for 3D viz
            distances = np.sum((X - center)**2, axis=1)
            rbf_features.append(np.exp(-gamma * distances))
        X_rbf_3d = np.column_stack(rbf_features)
        
        ax4.scatter(X_rbf_3d[:, 0], X_rbf_3d[:, 1], X_rbf_3d[:, 2], c=y, cmap='viridis', s=50)
        ax4.set_title('RBF Features\n(3 basis functions)', fontweight='bold')
        ax4.set_xlabel('RBF₁')
        ax4.set_ylabel('RBF₂')
        ax4.set_zlabel('RBF₃')
        
        # RBF Kernel PCA result
        ax5 = fig.add_subplot(2, 4, 5)
        kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=1.0)
        X_kpca_rbf = kpca_rbf.fit_transform(X)
        ax5.scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=y, cmap='viridis', s=50)
        ax5.set_title('RBF Kernel PCA\n(γ=1.0)', fontweight='bold')
        ax5.set_xlabel('PC1')
        ax5.set_ylabel('PC2')
        ax5.grid(True, alpha=0.3)
        
        # Standard PCA for comparison
        ax6 = fig.add_subplot(2, 4, 6)
        pca_standard = PCA(n_components=2)
        X_pca = pca_standard.fit_transform(X)
        ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
        ax6.set_title('Standard PCA\n(Linear)', fontweight='bold')
        ax6.set_xlabel('PC1')
        ax6.set_ylabel('PC2')
        ax6.grid(True, alpha=0.3)
        
        # Add explanation text
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.axis('off')
        explanation = """
🔍 Key Observations:

• Standard PCA fails to separate 
  the circular pattern

• Polynomial kernel maps to 
  higher dimensions where 
  separation becomes possible

• RBF kernel creates infinite-
  dimensional feature space

• Both kernel methods achieve
  perfect separation!

🎯 The Magic:
Non-linear data → Linear in 
higher dimensions → Perfect 
separation after projection
        """
        ax7.text(0.1, 0.9, explanation, transform=ax7.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Mathematical formulation
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.axis('off')
        math_text = """
📐 Mathematical Steps:

1️⃣ Original: X ∈ ℝ²

2️⃣ Kernel mapping:
   φ: ℝ² → ℝᵈ (d >> 2)

3️⃣ Kernel matrix:
   K[i,j] = φ(xᵢ)ᵀφ(xⱼ)

4️⃣ Eigendecomposition:
   Kα = λα

5️⃣ Project to 2D:
   Y = Kα[:,:2]

✨ Result: Non-linear → Linear!
        """
        ax8.text(0.1, 0.9, math_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig

def main():
    """
    Main function to run the rotating 3D demo
    """
    print("🚀 Starting Rotating 3D Kernel PCA Demo")
    print("="*50)
    
    demo = RotatingKernelPCADemo()
    
    print("\n1️⃣ Creating rotating animation...")
    anim = demo.create_rotating_animation(save_gif=False)
    
    print("\n2️⃣ Creating comprehensive comparison...")
    demo.create_side_by_side_comparison()
    
    print("\n✅ Demo complete!")
    print("\n💡 What you just saw:")
    print("   • 2D circular data that's not linearly separable")
    print("   • Transformation to 3D polynomial feature space")
    print("   • How the circular pattern becomes separable in 3D")
    print("   • Final 2D projection with perfect separation")
    
    return anim

if __name__ == "__main__":
    main()
