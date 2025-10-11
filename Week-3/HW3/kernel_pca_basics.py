import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KernelPCAVisualizer:
    """
    A comprehensive class to demonstrate and visualize Kernel PCA
    """
    
    def __init__(self):
        self.data = None
        self.labels = None
        self.scaler = StandardScaler()
        
    def create_synthetic_data(self, dataset_type='circles', n_samples=300, noise=0.1):
        """
        Create synthetic datasets that benefit from kernel transformations
        """
        print(f"\n=== Creating {dataset_type} dataset ===")
        
        if dataset_type == 'circles':
            # Concentric circles - perfect for RBF kernel
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.3, random_state=42)
            print("Created concentric circles dataset - ideal for RBF kernel")
            
        elif dataset_type == 'moons':
            # Two moons - good for polynomial kernel
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
            print("Created two moons dataset - good for polynomial kernel")
            
        elif dataset_type == 'spiral':
            # Spiral data - challenging for linear methods
            t = np.linspace(0, 4*np.pi, n_samples//2)
            noise_factor = noise
            
            # First spiral
            x1 = t * np.cos(t) + np.random.normal(0, noise_factor, len(t))
            y1 = t * np.sin(t) + np.random.normal(0, noise_factor, len(t))
            
            # Second spiral (offset)
            x2 = -t * np.cos(t) + np.random.normal(0, noise_factor, len(t))
            y2 = -t * np.sin(t) + np.random.normal(0, noise_factor, len(t))
            
            X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
            y = np.hstack([np.zeros(len(t)), np.ones(len(t))])
            print("Created spiral dataset - very challenging for linear methods")
            
        # Standardize the data
        X = self.scaler.fit_transform(X)
        self.data = X
        self.labels = y
        
        return X, y
    
    def visualize_original_data(self):
        """
        Visualize the original 2D data
        """
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.data[:, 0], self.data[:, 1], 
                            c=self.labels, cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter)
        plt.title('Original 2D Data', fontsize=14, fontweight='bold')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def apply_kernel_transformation(self, kernel_type='rbf', gamma=1.0, degree=3):
        """
        Apply kernel transformation and visualize the process
        """
        print(f"\n=== Applying {kernel_type.upper()} Kernel Transformation ===")
        
        if kernel_type == 'rbf':
            print(f"RBF Kernel: K(x,y) = exp(-γ||x-y||²) with γ={gamma}")
            print("Maps data to infinite-dimensional space")
            
        elif kernel_type == 'poly':
            print(f"Polynomial Kernel: K(x,y) = (γ<x,y> + 1)^d with γ={gamma}, d={degree}")
            print(f"Maps data to {degree}-degree polynomial feature space")
            
        elif kernel_type == 'linear':
            print("Linear Kernel: K(x,y) = <x,y>")
            print("Equivalent to standard PCA")
        
        # For visualization purposes, let's show what RBF kernel matrix looks like
        if kernel_type == 'rbf':
            self._visualize_kernel_matrix(gamma)
    
    def _visualize_kernel_matrix(self, gamma):
        """
        Visualize the RBF kernel matrix to show similarity relationships
        """
        # Compute RBF kernel matrix
        n_samples = len(self.data)
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                diff = self.data[i] - self.data[j]
                K[i, j] = np.exp(-gamma * np.dot(diff, diff))
        
        plt.figure(figsize=(10, 4))
        
        # Plot kernel matrix
        plt.subplot(1, 2, 1)
        plt.imshow(K, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'RBF Kernel Matrix (γ={gamma})')
        plt.xlabel('Sample Index')
        plt.ylabel('Sample Index')
        
        # Plot kernel values for a specific point
        plt.subplot(1, 2, 2)
        center_idx = len(self.data) // 2
        kernel_values = K[center_idx, :]
        scatter = plt.scatter(self.data[:, 0], self.data[:, 1], 
                            c=kernel_values, cmap='viridis', s=50)
        plt.colorbar(scatter)
        plt.title(f'Kernel Similarities to Point {center_idx}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()
    
    def compare_pca_methods(self, n_components=2):
        """
        Compare standard PCA with different Kernel PCA variants
        """
        print(f"\n=== Comparing PCA Methods ===")
        
        # Standard PCA
        pca_standard = PCA(n_components=n_components)
        X_pca = pca_standard.fit_transform(self.data)
        
        # Kernel PCA variants
        kernels = {
            'Linear': KernelPCA(n_components=n_components, kernel='linear'),
            'RBF (γ=1)': KernelPCA(n_components=n_components, kernel='rbf', gamma=1.0),
            'RBF (γ=5)': KernelPCA(n_components=n_components, kernel='rbf', gamma=5.0),
            'Polynomial (d=2)': KernelPCA(n_components=n_components, kernel='poly', degree=2),
            'Polynomial (d=3)': KernelPCA(n_components=n_components, kernel='poly', degree=3)
        }
        
        # Apply transformations
        results = {'Standard PCA': X_pca}
        for name, kpca in kernels.items():
            X_transformed = kpca.fit_transform(self.data)
            results[name] = X_transformed
        
        # Visualize results
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (method_name, X_transformed) in enumerate(results.items()):
            ax = axes[idx]
            scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                               c=self.labels, cmap='viridis', alpha=0.7, s=50)
            ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('First Component')
            ax.set_ylabel('Second Component')
            ax.grid(True, alpha=0.3)
            
            # Add explained variance for standard PCA
            if method_name == 'Standard PCA':
                var_ratio = pca_standard.explained_variance_ratio_
                ax.text(0.02, 0.98, f'Explained Variance:\nPC1: {var_ratio[0]:.2f}\nPC2: {var_ratio[1]:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def demonstrate_kernel_benefits(self):
        """
        Demonstrate why kernel methods are better for non-linear data
        """
        print("\n=== Why Kernel PCA is Better ===")
        print("1. **Linear PCA Limitations:**")
        print("   - Only finds linear combinations of original features")
        print("   - Cannot capture non-linear patterns")
        print("   - Poor separation for curved/circular patterns")
        
        print("\n2. **RBF Kernel Benefits:**")
        print("   - Maps data to infinite-dimensional space")
        print("   - Captures local similarities between points")
        print("   - Excellent for circular/radial patterns")
        print("   - γ parameter controls locality (higher γ = more local)")
        
        print("\n3. **Polynomial Kernel Benefits:**")
        print("   - Maps to polynomial feature space")
        print("   - Captures polynomial relationships")
        print("   - Good for curved boundaries")
        print("   - Degree parameter controls complexity")
        
        # Show separation quality
        self._analyze_separation_quality()
    
    def _analyze_separation_quality(self):
        """
        Analyze how well different methods separate the classes
        """
        print("\n=== Separation Quality Analysis ===")
        
        # Apply different methods
        pca_standard = PCA(n_components=2)
        X_pca = pca_standard.fit_transform(self.data)
        
        kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=1.0)
        X_kpca_rbf = kpca_rbf.fit_transform(self.data)
        
        kpca_poly = KernelPCA(n_components=2, kernel='poly', degree=3)
        X_kpca_poly = kpca_poly.fit_transform(self.data)
        
        # Calculate separation metrics (simplified)
        methods = {
            'Standard PCA': X_pca,
            'Kernel PCA (RBF)': X_kpca_rbf,
            'Kernel PCA (Poly)': X_kpca_poly
        }
        
        plt.figure(figsize=(15, 5))
        
        for idx, (name, X_transformed) in enumerate(methods.items()):
            plt.subplot(1, 3, idx+1)
            
            # Plot the transformed data
            scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                                c=self.labels, cmap='viridis', alpha=0.7, s=50)
            plt.title(f'{name}', fontsize=12, fontweight='bold')
            plt.xlabel('First Component')
            plt.ylabel('Second Component')
            plt.grid(True, alpha=0.3)
            
            # Calculate and display class separation
            class_0 = X_transformed[self.labels == 0]
            class_1 = X_transformed[self.labels == 1]
            
            center_0 = np.mean(class_0, axis=0)
            center_1 = np.mean(class_1, axis=0)
            separation = np.linalg.norm(center_1 - center_0)
            
            plt.text(0.02, 0.02, f'Class Separation: {separation:.2f}', 
                    transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to demonstrate Kernel PCA concepts
    """
    print("=" * 60)
    print("KERNEL PCA DEMONSTRATION AND VISUALIZATION")
    print("=" * 60)
    
    visualizer = KernelPCAVisualizer()
    
    # Test different datasets
    datasets = ['circles', 'moons', 'spiral']
    
    for dataset in datasets:
        print(f"\n{'='*20} {dataset.upper()} DATASET {'='*20}")
        
        # Create and visualize data
        X, y = visualizer.create_synthetic_data(dataset)
        visualizer.visualize_original_data()
        
        # Show kernel transformations
        visualizer.apply_kernel_transformation('rbf', gamma=1.0)
        
        # Compare different PCA methods
        results = visualizer.compare_pca_methods()
        
        # Demonstrate benefits
        visualizer.demonstrate_kernel_benefits()
        
        print(f"\nCompleted analysis for {dataset} dataset!")
        print("-" * 50)

if __name__ == "__main__":
    main()