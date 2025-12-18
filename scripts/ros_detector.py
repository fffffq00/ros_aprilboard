#!/usr/bin/env python3
import sys
from pathlib import Path
__package__ = Path(__file__).parent.name
sys.path.append(str(Path(__file__).parent.parent))


import sys
from pathlib import Path
pkg_root = Path(__file__).parent
sys.path.insert(0, str(pkg_root))

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped
from visualization_msgs.msg import Marker
import tf   

from aprilboard import AprilDetector, AprilBoard


def solve_pnp_ippe_select(objpts, imgpts, K, D):
    retvals, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
        objpts, imgpts, K, D, flags=cv2.SOLVEPNP_IPPE
    )
    if not retvals or len(rvecs) == 0:
        return False, None, None, None
    # solvePnPGeneric 返回的是 list
    reproj_errs = np.array(reproj_errs).reshape(-1)
    best_idx = int(np.argmin(reproj_errs))
    best_rvec = rvecs[best_idx]
    best_tvec = tvecs[best_idx]
    best_error = float(reproj_errs[best_idx])

    return True, best_rvec, best_tvec, best_error


class RosAprilboardDetectorNode:
    def __init__(self):
        # params for board
        rows = rospy.get_param("~rows", 4)
        cols = rospy.get_param("~cols", 4)
        marker_length = rospy.get_param("~marker_length", 0.05)  # meters
        space_length = rospy.get_param("~space_length", marker_length / 2)
        start_id = rospy.get_param("~start_id", 0)

        # frames
        self.camera_frame = rospy.get_param("~camera_frame", "camera_rgb_optical_frame")
        self.reference_frame = rospy.get_param("~reference_frame", self.camera_frame)
        self.marker_frame = rospy.get_param("~marker_frame", "board")
        self.tag_family_name = rospy.get_param("~tag_family", "t36h11")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera_info")
        self.camera_image_topic = rospy.get_param("~camera_image_topic", "/image_rect_color")

        self.draw_markers = rospy.get_param("~draw_markers", True)
        self.draw_corners = rospy.get_param("~draw_corners", False)

        # create board and detector
        self.board = AprilBoard(rows, cols, marker_length, space_length, start_id, self.tag_family_name)
        self.detector = AprilDetector(self.board)

        # publishers
        self.image_pub = rospy.Publisher("result", Image, queue_size=1)
        self.debug_pub = rospy.Publisher("debug", Image, queue_size=1)
        self.pose_pub = rospy.Publisher("pose", PoseStamped, queue_size=10)

        # TF
        self.tf_broadcaster = tf.TransformBroadcaster()

        # camera params
        self.cam_K = None
        self.cam_D = None
        self.cam_info_received = False

        self.bridge = CvBridge()

        # subscribers
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
        rospy.Subscriber(self.camera_image_topic, Image, self.image_callback, queue_size=1)

        rospy.loginfo("ros_aprilboard_detector_node started")

    def cam_info_callback(self, msg: CameraInfo):
        if self.cam_info_received:
            return
        K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.D, dtype=np.float64)
        self.cam_K = K
        self.cam_D = D
        self.cam_info_received = True
        rospy.loginfo("CameraInfo received")

    def image_callback(self, msg: Image):
        if not self.cam_info_received:
            rospy.logdebug("No camera info yet")
            return

        # avoid work if nobody subscribed to any outputs
        if (self.image_pub.get_num_connections() == 0 and self.debug_pub.get_num_connections() == 0
                and self.pose_pub.get_num_connections() == 0):
            rospy.logdebug("No subscribers, skipping detection")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("cv_bridge error: %s", e)
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # detect
        corners, ids = self.detector.detectBoard(gray)

        if len(corners) == 0:
            # 当没有检测到任何标记时，发布原始图像
            if self.image_pub.get_num_connections() > 0:
                try:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8'))
                except CvBridgeError:
                    pass
            # 仍然发布调试阈值图像（如果有订阅者）
            if self.debug_pub.get_num_connections() > 0:
                try:
                    thresh = self.detector.threshold(gray)
                    self.debug_pub.publish(self.bridge.cv2_to_imgmsg(thresh, encoding='mono8'))
                except CvBridgeError:
                    pass
            return
        

        # draw detected markers on image
        if self.draw_markers or self.draw_corners: 
            draw_marker_corners = []
            draw_marker_ids = []
            for id, corner in zip(ids, corners):
                if self.draw_markers:
                    draw_marker_corners.append(corner.reshape(1,4,2))
                    draw_marker_ids.append(np.array([id]))
                if self.draw_corners:
                    for j, c in enumerate(corner):
                        p = np.round(c).astype(np.int32)
                        cv2.circle(cv_image, tuple(p), 3, (255, 0, 0), -1)
            if self.draw_markers:
                cv2.aruco.drawDetectedMarkers(cv_image, draw_marker_corners, np.array(draw_marker_ids))
            

        # try to estimate pose of whole board using matched points
        objpts, imgpts = self.board.matchImagePoints(corners, ids)

        # ensure shapes
        objpts = objpts.reshape(-1, 3).astype(np.float64)
        imgpts = imgpts.reshape(-1, 2).astype(np.float64)
        retval, rvec, tvec, reperror = solve_pnp_ippe_select(objpts, imgpts, self.cam_K, self.cam_D)
        

        if not retval:
            rospy.logdebug("solvePnP returned false")
            return
        
        rospy.loginfo(f"IPPE ReProjectError {reperror}")

        cv2.drawFrameAxes(
            cv_image, self.cam_K, self.cam_D, rvec, tvec, self.board.markerLength * 0.5
        )

        # convert to quaternion
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        quat = tf.transformations.quaternion_from_matrix(T)
        t = (float(tvec[0]), float(tvec[1]), float(tvec[2]))

        # publish TF (child = marker_frame, parent = reference_frame)
        now = rospy.Time.now()
        self.tf_broadcaster.sendTransform(t, quat, now, self.marker_frame, self.reference_frame)

        # publish PoseStamped
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = self.reference_frame
        pose.pose.position.x = t[0]
        pose.pose.position.y = t[1]
        pose.pose.position.z = t[2]
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose)


        # publish annotated image
        if self.image_pub.get_num_connections() > 0:
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8'))
            except CvBridgeError:
                pass

        # publish debug threshold image
        if self.debug_pub.get_num_connections() > 0:
            try:
                thresh = self.detector.threshold(gray)
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(thresh, encoding='mono8'))
            except CvBridgeError:
                pass

if __name__ == "__main__":
    rospy.init_node("ros_aprilboard_detector_node", anonymous=True)
    node = RosAprilboardDetectorNode()
    rospy.spin()