    def image_callback(self, msg):
        if not self.camera_initialized:
            self.get_logger().debug("⏳ 摄像头尚未完全初始化，跳过本次回调")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            results = self.model(cv_image, conf=0.6)[0]
            current_detections = []
            
            current_pose = self.path_history[-1] if self.path_history else (None, None)
            x_pose, y_pose = current_pose
            
            min_object_distance = float('inf')
            has_close_object = False
            
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result.cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                depth = self.get_depth_at_point(cx, cy)
                
                obj = {
                    "物件": self.model.names[int(cls)],
                    "信心分數": round(float(conf), 2),
                    "距離(m)": round(float(depth), 2),
                    "x": round(float(x_pose), 2) if x_pose is not None else None,
                    "y": round(float(y_pose), 2) if y_pose is not None else None
                }
                current_detections.append(obj)
                
                # Check if object is closer than 1.0 meter
                if depth > 0 and depth < 1.0:
                    has_close_object = True
                    min_object_distance = min(min_object_distance, depth)
            
            self.detections_for_ui = current_detections
            
            # --- OBSTACLE HANDLING LOGIC ---
            if self.navigation_active and has_close_object:
                
                # 1. FORCE ROBOT TO STOP (Prevents it from going around)
                # We publish 0 velocity constantly as long as the object is < 1.0m
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_publisher.publish(twist)
                
                # 2. Trigger warnings and UI updates
                current_time = time.time()
                if (current_time - self.last_warning_time > self.warning_cooldown and 
                    not self.warning_speech_active):
                    self.play_warning_speech()
                    self.last_warning_time = current_time
                    self.get_logger().warn(f"🛑 紧急停止！YOLO物體距離警告：{min_object_distance:.2f}m")
                    
                    if not self.is_avoiding_obstacle:
                        self.pause_normal_speech()
                        self.is_avoiding_obstacle = True
            else:
                # Once object is out of the way, resume normal behavior
                if self.is_avoiding_obstacle:
                    self.resume_normal_speech()
                    self.is_avoiding_obstacle = False
            
            # --- SEGMENTATION ---
            seg_results = self.seg_model(cv_image, conf=0.5)[0]
            segmented_image = self.draw_segmentation(cv_image, seg_results)
            
            with self.segmentation_lock:
                self.latest_segmented_image = segmented_image
                
        except Exception as e:
            self.get_logger().error(f'影像處理錯誤: {str(e)}')
