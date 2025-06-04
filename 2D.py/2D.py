import pygame
from pygame.math import Vector2

# 1. 建立骨骼結構
class Bone:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.position = Vector2(0, 0)  # 2D位置
        self.rotation = 0  # 旋轉角度
        self.children = []

# 2. 定義關鍵幀
class KeyFrame:
    def __init__(self, time, position, rotation):
        self.time = time
        self.position = position 
        self.rotation = rotation

# 3. 動畫系統
class SkeletalAnimation:
    def __init__(self):
        self.bones = {}  # 存儲所有骨骼
        self.keyframes = {}  # 存儲每個骨骼的關鍵幀
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.running = True
        
    def add_bone(self, bone):
        """添加骨骼到系統"""
        self.bones[bone.name] = bone
        
    def add_keyframe(self, bone_name, keyframe):
        """添加關鍵幀到指定骨骼"""
        if bone_name not in self.keyframes:
            self.keyframes[bone_name] = []
        self.keyframes[bone_name].append(keyframe)
    
    def interpolate_keyframes(self, bone_name, current_time):
        """在關鍵幀之間進行插值計算"""
        if bone_name not in self.keyframes:
            return Vector2(0, 0), 0
            
        keyframes = self.keyframes[bone_name]
        if not keyframes:
            return Vector2(0, 0), 0
            
        # 處理循環動畫
        total_time = keyframes[-1].time
        current_time = current_time % total_time  # 使動畫循環
            
        # 找到當前時間的前後關鍵幀
        prev_frame = keyframes[0]
        next_frame = keyframes[-1]
        
        for i in range(len(keyframes) - 1):
            if keyframes[i].time <= current_time <= keyframes[i + 1].time:
                prev_frame = keyframes[i]
                next_frame = keyframes[i + 1]
                break
                
        # 線性插值
        if prev_frame == next_frame:
            return prev_frame.position, prev_frame.rotation
            
        # 確保t值在0到1之間
        t = max(0, min(1, (current_time - prev_frame.time) / (next_frame.time - prev_frame.time)))
        pos = prev_frame.position.lerp(next_frame.position, t)
        rot = prev_frame.rotation + (next_frame.rotation - prev_frame.rotation) * t
        
        return pos, rot
    
    def update(self, current_time):
        """更新所有骨骼的位置和旋轉"""
        for bone_name in self.bones:
            position, rotation = self.interpolate_keyframes(bone_name, current_time)
            self.bones[bone_name].position = position
            self.bones[bone_name].rotation = rotation
    
    def draw(self):
        """繪製骨骼"""
        self.screen.fill((0, 0, 0))  # 清空畫面
        
        # 繪製每個骨骼
        for bone in self.bones.values():
            # 繪製骨骼點
            pygame.draw.circle(self.screen, (0, 255, 255), 
                             (int(bone.position.x), int(bone.position.y)), 5)
            
            # 如果有父骨骼，繪製連接線
            if bone.parent:
                pygame.draw.line(self.screen, (255, 255, 255),
                               (bone.position.x, bone.position.y),
                               (bone.parent.position.x, bone.parent.position.y), 2)
        
        pygame.display.flip()
    
    def run(self):
        """主循環"""
        current_time = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            self.update(current_time)
            self.draw()
            current_time += 0.016  # 約60FPS
            self.clock.tick(60)
        
        pygame.quit()

# 測試代碼
if __name__ == "__main__":
    # 創建動畫系統
    animation = SkeletalAnimation()
    
    # 創建骨骼結構
    root = Bone("root")
    root.position = Vector2(400, 300)  # 設置在畫面中心
    
    spine = Bone("spine", root)
    spine.position = Vector2(400, 250)
    
    # 添加骨骼到系統
    animation.add_bone(root)
    animation.add_bone(spine)
    
    # 添加關鍵幀
    animation.add_keyframe("root", KeyFrame(0, Vector2(400, 300), 0))
    animation.add_keyframe("root", KeyFrame(1, Vector2(400, 320), 45))
    animation.add_keyframe("root", KeyFrame(2, Vector2(400, 300), 0))
    
    # 運行動畫
    animation.run()
