// 完全自包含的PLY查看器类
class PLYViewer {
    constructor(containerId, options = {}) {
      // 合并默认配置
      this.config = {
        backgroundColor: 0x222222,
        pointSize: 0.05,
        defaultColor: 0x00aaff,
        autoRotate: false,
        rotateSpeed: 0.4,
        ...options
      };
  
      // 初始化Three.js核心组件
      this.initScene(containerId);
      this.initLights();
      this.initControls();
      this.startAnimationLoop();
  
      // 自动响应窗口大小变化
      window.addEventListener('resize', () => this.handleResize());
    }

  
    /* 私有方法 */
    initScene(containerId) {
      this.container = document.getElementById(containerId);

      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(this.config.backgroundColor);

      const w = this.container.clientWidth || 600;
      const h = this.container.clientHeight || 400;

      this.camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 1e7);
      this.camera.position.set(0, 0, 2);

      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      this.renderer.setSize(w, h);

      // 可选：清空容器避免重复 append canvas
      this.container.innerHTML = "";
      this.container.appendChild(this.renderer.domElement);
    }
  
    initLights() {
      this.scene.add(new THREE.AmbientLight(0x404040));
      const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
      directionalLight.position.set(1, 1, 1);
      this.scene.add(directionalLight);
    }
  
    initControls() {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      // this.controls.autoRotate = this.config.autoRotate;
      this.controls.autoRotate = false;
      this.controls.minPolarAngle = - Math.PI;      // 允许垂直旋转到0度（正下方）
      this.controls.maxPolarAngle = Math.PI; // 允许垂直旋转到180度（正上方）
      this.controls.maxDistance = 1000;    // 允许相机远离模型

      // ===== 新增：交互时暂停自旋转 =====
      this._userInteracting = false;

      this.controls.addEventListener('start', () => {
        this._userInteracting = true;   // 开始交互 -> 暂停
      });

      this.controls.addEventListener('end', () => {
        this._userInteracting = false;  // 结束交互 -> 恢复
      });
    }
  
    startAnimationLoop() {
      const clock = new THREE.Clock();

      const animate = () => {
        requestAnimationFrame(animate);

        const dt = clock.getDelta();

        // 关键：沿 Z 轴自旋转
        if (this.config.autoRotate && this.modelRoot && !this._userInteracting) {
          this.modelRoot.rotation.z += this.config.rotateSpeed * dt;
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
      };

      animate();
    }
  
    handleResize() {
      if (!this.container) return;
      const w = this.container.clientWidth || 600;
      const h = this.container.clientHeight || 400;

      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    }

    frameObject(object3d) {
      const box = new THREE.Box3().setFromObject(object3d);
      if (box.isEmpty()) return;

      // const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;

      // // 1) 把模型移到原点（让模型“居中”）
      // object3d.position.sub(center);

      // 2) 让 OrbitControls 围绕原点旋转
      this.controls.target.set(0, 0, 0);
      this.controls.update();

      // 3) 根据 FOV 计算合适距离，让模型大小合适
      const fov = THREE.MathUtils.degToRad(this.camera.fov);
      const fitDist = (maxDim * 0.5) / Math.tan(fov * 0.5);
      const dist = fitDist * 1.3; // 1.3 是留边

      this.camera.position.set(0, 0, dist);
      this.camera.near = dist / 1000;
      this.camera.far = dist * 1000;
      this.camera.updateProjectionMatrix();

      this.controls.maxDistance = dist * 10;
    }

    async loadPLY(modelPath, onProgress) {
      return new Promise((resolve, reject) => {
        new THREE.PLYLoader().load(
          modelPath,
          (geometry) => {
            // 清理旧模型（可选，但建议）
            this.clearScene();

            geometry.computeBoundingBox();
            const center = geometry.boundingBox.getCenter(new THREE.Vector3());
            geometry.translate(-center.x, -center.y, -center.z);

            // 更新包围盒/球，供取景使用
            geometry.computeBoundingBox();
            geometry.computeBoundingSphere();

            // 点云
            const pointsMaterial = new THREE.PointsMaterial({
              size: this.config.pointSize,
              vertexColors: geometry.hasAttribute('color'),
              color: this.config.defaultColor
            });
            const points = new THREE.Points(geometry, pointsMaterial);

            // 线段（注意：你的 PLYLoader 不一定真的带 edge index）
            const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
            const lines = new THREE.LineSegments(geometry, lineMaterial);

            // 放进同一个 group，统一居中/取景
            const group = new THREE.Group();
            group.add(points);
            group.add(lines);
            this.scene.add(group);

            // 关键：保存根节点，给动画循环用
            this.modelRoot = group;

            // 关键：居中 + 合适取景 + 设置旋转中心
            this.frameObject(group);

            resolve({ group, points, lines });
          },
          onProgress,
          (error) => reject(error)
        );
      });
    }

    clearScene() {
      if (this.modelRoot) {
        this.scene.remove(this.modelRoot);
        this.modelRoot.traverse(obj => {
          if (obj.geometry) obj.geometry.dispose();
          if (obj.material) {
            if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
            else obj.material.dispose();
          }
        });
        this.modelRoot = null;
      }
    }
  }
