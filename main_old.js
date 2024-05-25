// main.js
document.addEventListener('DOMContentLoaded', function () {
    // container
    const container = document.getElementById('container');

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    // const renderer = new THREE.WebGLRenderer();
    const renderer = new THREE.WebGLRenderer({ canvas: rendererCanvas });
    renderer.setSize(window.innerWidth * 0.5, window.innerHeight * 0.5);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.screenSpacePanning = false;
    controls.maxPolarAngle = Math.PI / 2;

    // 设置控制器的样式
    // const controlsElement = controls.domElement;
    // controlsElement.style.position = 'absolute';
    // controlsElement.style.top = '10px';
    // controlsElement.style.left = '10px';
    // controlsElement.style.zIndex = '100';
    // container.appendChild(controlsElement); // 将控制器添加到容器中而不是渲染器的父元素
    // container.appendChild(controls.domElement); // 将控制器添加到容器中而不是渲染器的父元素

    const loader = new THREE.PCDLoader();
    let pointCloud;

    loader.load('pcd/Luca_Biada_flickr_2.pcd', function (points) {
        // 获取点云的材质
        const material = points.material;
        // 设置点云的初始颗粒大小
        material.size = 0.05;
        pointCloud = points;
        // pointCloud = new THREE.Points(points, material); // 使用新的点云材质创建点云对象
        scene.add(pointCloud);
        const center = points.geometry.boundingSphere.center;
        controls.target.set(center.x, center.y, center.z);
        controls.update();
        camera.position.set(center.x, center.y, center.z + 5);
        animate();
    });

    window.addEventListener('resize', function () {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth * 0.5, window.innerHeight * 0.5);
    });

    document.getElementById('scale').addEventListener('input', function (event) {
        const scale = event.target.value;
        if (pointCloud) {
            pointCloud.scale.set(scale, scale, scale);
        }
    });

    document.getElementById('posX').addEventListener('input', function (event) {
        const posX = event.target.value;
        if (pointCloud) {
            pointCloud.position.x = posX;
        }
    });

    document.getElementById('posY').addEventListener('input', function (event) {
        const posY = event.target.value;
        if (pointCloud) {
            pointCloud.position.y = posY;
        }
    });

    document.getElementById('posZ').addEventListener('input', function (event) {
        const posZ = event.target.value;
        if (pointCloud) {
            pointCloud.position.z = posZ;
        }
    });
    // 滚动条事件监听器来调整点云颗粒大小
    document.getElementById('pointSize').addEventListener('input', function (event) {
        const pointSize = event.target.value; // 获取滚动条的值
        if (pointCloud) {
            pointCloud.material.size = pointSize; // 更新点云材质的颗粒大小
        }
    });

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
});
