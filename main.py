import taichi as ti
import numpy as np

# 初始化 Taichi，使用 GPU 加速渲染
ti.init(arch=ti.gpu)

# 全局配置参数
WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000 # 贝塞尔曲线的采样数
# B 样条因为是分段的，如果控制点满了(100个)，最多有97段，每段采100个点，最多近10000个点
MAX_CURVE_POINTS = 10000 

# 显存缓冲区分配
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)

# ==========================================
# 算法 1：纯 Python 递归实现 De Casteljau (贝塞尔)
# ==========================================
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

# ==========================================
# 算法 2：矩阵形式计算均匀三次 B 样条分段
# ==========================================
# 固定的三次 B 样条基矩阵
M_bspline = np.array([
    [-1,  3, -3,  1],
    [ 3, -6,  3,  0],
    [-3,  0,  3,  0],
    [ 1,  4,  1,  0]
]) / 6.0

def compute_bspline_segment(p0, p1, p2, p3, num_samples=100):
    # 构造控制点矩阵 (4x2)
    P = np.array([p0, p1, p2, p3]) 
    T = np.zeros((num_samples, 4))
    
    # 生成参数矩阵 T (num_samples x 4)
    for i in range(num_samples):
        t = i / (num_samples - 1)
        T[i] = [t**3, t**2, t, 1]
        
    # 矩阵连乘：T @ M @ P 得到当前段在曲线上所有的采样点坐标 (num_samples x 2)
    return T @ M_bspline @ P

# ==========================================
# GPU 内核逻辑
# ==========================================
@ti.kernel
def clear_pixels():
    """并行清空屏幕"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32, color_channel: ti.i32):
    """
    带反走样（抗锯齿）的光栅化绘制内核
    """
    for i in range(n):
        pt = curve_points_field[i]
        # 获得精确的亚像素级浮点坐标
        xf = pt[0] * WIDTH
        yf = pt[1] * HEIGHT
        
        # 获得中心整数坐标
        xi = ti.cast(xf, ti.i32)
        yi = ti.cast(yf, ti.i32)
        
        # 考察 3x3 局部像素邻域
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px = xi + dx
                py = yi + dy
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    # 计算该物理像素中心与精确几何点之间的欧氏距离
                    dist = ti.math.sqrt((px - xf)**2 + (py - yf)**2)
                    
                    # 距离衰减模型：距离越近，权重越大。距离大于 1 则无贡献
                    weight = ti.max(0.0, 1.0 - dist)
                    
                    # 使用 atomic_max 确保同一个像素被多个密集采样点覆盖时，不会发生累加过曝
                    ti.atomic_max(pixels[px, py][color_channel], weight)

# ==========================================
# 主循环与交互
# ==========================================
def main():
    window = ti.ui.Window("Bezier vs B-Spline (Anti-Aliased)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    # 状态机：'bezier' 或 'bspline'
    mode = 'bezier' 
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB: 
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
            elif e.key == 'c': 
                control_points = []
            elif e.key == 'b':
                # 切换绘制模式
                mode = 'bspline' if mode == 'bezier' else 'bezier'
                print(f"Switched to {mode.upper()} mode")

        clear_pixels()
        current_count = len(control_points)
        
        # ---------------- 贝塞尔曲线逻辑 ----------------
        if mode == 'bezier' and current_count >= 2:
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)
                
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(NUM_SEGMENTS + 1, 1) # 通道 1 为绿色
            
        # ---------------- B 样条曲线逻辑 ----------------
        elif mode == 'bspline' and current_count >= 4:
            # 每 4 个点构成一段三次 B 样条，总共 n-3 段
            num_segments = current_count - 3
            samples_per_seg = 100
            total_points = num_segments * samples_per_seg
            
            curve_points_np = np.zeros((total_points, 2), dtype=np.float32)
            idx = 0
            for i in range(num_segments):
                # 截取局部的 4 个控制点
                pts = control_points[i:i+4]
                seg_points = compute_bspline_segment(*pts, num_samples=samples_per_seg)
                curve_points_np[idx:idx+samples_per_seg] = seg_points
                idx += samples_per_seg
                
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(total_points, 2) # 通道 2 为蓝色，以示区分

        # 将显存映射到屏幕
        canvas.set_image(pixels)
        
        # 绘制交互控制点（对象池技巧）
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        window.show()

if __name__ == '__main__':
    main()