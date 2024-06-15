// main.js
document.addEventListener('DOMContentLoaded', function () {
    // const container = document.getElementById('container');
    // const rendererCanvas = document.getElementById('renderer');
    // const scene = new THREE.Scene();
    // const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    // const renderer = new THREE.WebGLRenderer({ canvas: rendererCanvas });
    // renderer.setSize(container.offsetWidth, container.offsetHeight);

    // const controls = new THREE.OrbitControls(camera, renderer.domElement);
    // controls.enableDamping = true;
    // controls.dampingFactor = 0.25;
    // controls.screenSpacePanning = false;
    // controls.maxPolarAngle = Math.PI / 2;

    // const loader = new THREE.PCDLoader();
    // let pointCloud;

    // loader.load('pcd/Luca_Biada_flickr_2.pcd', function (points) {
    //     const material = points.material;
    //     material.size = 0.05;
    //     pointCloud = points;
    //     scene.add(pointCloud);
    //     const center = points.geometry.boundingSphere.center;
    //     controls.target.set(center.x, center.y, center.z);
    //     controls.update();
    //     camera.position.set(center.x, center.y, center.z + 5);
    //     animate();
    // });

    // window.addEventListener('resize', function () {
    //     camera.aspect = container.offsetWidth / container.offsetHeight;
    //     camera.updateProjectionMatrix();
    //     renderer.setSize(container.offsetWidth, container.offsetHeight);
    // });

    function init(containerId, controlsId, pcdFile) {
        const container = document.getElementById(containerId);
        const controlsElement = document.getElementById(controlsId);
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, container.offsetWidth / container.offsetHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(container.offsetWidth, container.offsetHeight);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        controls.screenSpacePanning = false;
        controls.maxPolarAngle = Math.PI / 2;

        const loader = new THREE.PCDLoader();
        let pointCloud;

        loader.load(pcdFile, function (points) {
            points.material.size = 0.005;
            pointCloud = points;
            scene.add(pointCloud);
            const center = points.geometry.boundingSphere.center;
            controls.target.set(center.x, center.y, center.z);
            controls.update();
            camera.position.set(center.x, center.y, center.z + 10);
            animate();
        });

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        window.addEventListener('resize', function () {
            camera.aspect = container.offsetWidth / container.offsetHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.offsetWidth, container.offsetHeight);
        });

        controlsElement.querySelector(`#${controlsId} input[name="scale"]`).addEventListener('input', function (event) {
            const scale = event.target.value;
            if (pointCloud) {
                pointCloud.scale.set(scale, scale, scale);
            }
        });

        controlsElement.querySelector(`#${controlsId} input[name="posX"]`).addEventListener('input', function (event) {
            const posX = event.target.value;
            if (pointCloud) {
                pointCloud.position.x = posX;
            }
        });

        controlsElement.querySelector(`#${controlsId} input[name="posY"]`).addEventListener('input', function (event) {
            const posY = event.target.value;
            if (pointCloud) {
                pointCloud.position.y = posY;
            }
        });

        controlsElement.querySelector(`#${controlsId} input[name="posZ"]`).addEventListener('input', function (event) {
            const posZ = event.target.value;
            if (pointCloud) {
                pointCloud.position.z = posZ;
            }
        });

        controlsElement.querySelector(`#${controlsId} input[name="pointSize"]`).addEventListener('input', function (event) {
            const pointSize = event.target.value;
            if (pointCloud) {
                pointCloud.material.size = pointSize;
            }
        });
    }

    // 初始化两个不同的点云
    init('container1547', 'controls1547', 'pcd/1547_rgb.pcd');
    init('container1556', 'controls1556', 'pcd/bifuse_1556_rgb.pcd');
    init('container1', 'controls1', 'pcd/Luca_Biada_flickr_2.pcd');
    init('container2', 'controls2', 'pcd/Luca_Biada_flickr_trainstation.pcd');

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

    document.getElementById('pointSize').addEventListener('input', function (event) {
        const pointSize = event.target.value;
        if (pointCloud) {
            pointCloud.material.size = pointSize;
        }
    });

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
});
