import numpy as np
import cv2

# Define the control points
control_points = np.array([
    [100, 100],
    [0, 160],
    [200, 260],
    [110, 330]
], dtype=np.float32)

def binomial_coeffs(n):
    """Compute binomial coefficients for given n."""
    C = np.zeros(n + 1)
    C[0] = 1
    for i in range(1, n + 1):
        C[i] = 1
        for j in range(i - 1, 0, -1):
            C[j] += C[j - 1]
    return C

def compute_bez_pt(u, control_points, C):
    bez_pt = np.zeros(2, dtype=np.float64) 
    n = len(control_points) - 1
    for k in range(n + 1):
        bez_blend_fn = C[k] * (u ** k) * ((1 - u) ** (n - k))
        bez_pt += bez_blend_fn * np.array(control_points[k], dtype=np.float64)  
    return bez_pt

def bezier(control_points, num_points=1000):
    """Draw the Bezier curve using the control points."""
    n = len(control_points) - 1
    C = binomial_coeffs(n)
    bezier_points = np.array([compute_bez_pt(u, control_points, C) for u in np.linspace(0, 1, num_points)])
    return bezier_points

def plot_bezier_curve(control_points):
    """Plot the Bezier curve and control points using OpenCV."""
    bezier_points = bezier(control_points)
    
    # Create a blank image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Draw the Bezier curve
    for i in range(len(bezier_points) - 1):
        pt1 = tuple(bezier_points[i].astype(int))
        pt2 = tuple(bezier_points[i + 1].astype(int))
        cv2.line(img, pt1, pt2, (255, 255, 255), 2)
    
    for i in range(len(control_points) - 1):
        pt1 = tuple(control_points[i].astype(int))
        pt2 = tuple(control_points[i + 1].astype(int))
        cv2.line(img, pt1, pt2, (0, 255, 0), 1) 
        cv2.circle(img, pt1, 5, (0, 0, 255), -1)  
    cv2.circle(img, tuple(control_points[-1].astype(int)), 5, (0, 0, 255), -1)  
    
    # Display the image
    cv2.imshow('Bezier Curve', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Plot the Bezier curve
#plot_bezier_curve(control_points)