;; Auto-generated. Do not edit!


(when (boundp 'gazebo_radiation_plugins::MassYamlLoader)
  (if (not (find-package "GAZEBO_RADIATION_PLUGINS"))
    (make-package "GAZEBO_RADIATION_PLUGINS"))
  (shadow 'MassYamlLoader (find-package "GAZEBO_RADIATION_PLUGINS")))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::MASSYAMLLOADER")
  (make-package "GAZEBO_RADIATION_PLUGINS::MASSYAMLLOADER"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::MASSYAMLLOADERREQUEST")
  (make-package "GAZEBO_RADIATION_PLUGINS::MASSYAMLLOADERREQUEST"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::MASSYAMLLOADERRESPONSE")
  (make-package "GAZEBO_RADIATION_PLUGINS::MASSYAMLLOADERRESPONSE"))

(in-package "ROS")





(defclass gazebo_radiation_plugins::MassYamlLoaderRequest
  :super ros::object
  :slots (_folder ))

(defmethod gazebo_radiation_plugins::MassYamlLoaderRequest
  (:init
   (&key
    ((:folder __folder) "")
    )
   (send-super :init)
   (setq _folder (string __folder))
   self)
  (:folder
   (&optional __folder)
   (if __folder (setq _folder __folder)) _folder)
  (:serialization-length
   ()
   (+
    ;; string _folder
    4 (length _folder)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _folder
       (write-long (length _folder) s) (princ _folder s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _folder
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _folder (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass gazebo_radiation_plugins::MassYamlLoaderResponse
  :super ros::object
  :slots ())

(defmethod gazebo_radiation_plugins::MassYamlLoaderResponse
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass gazebo_radiation_plugins::MassYamlLoader
  :super ros::object
  :slots ())

(setf (get gazebo_radiation_plugins::MassYamlLoader :md5sum-) "bf1b291db59c399c5c5b25859bd3e2e3")
(setf (get gazebo_radiation_plugins::MassYamlLoader :datatype-) "gazebo_radiation_plugins/MassYamlLoader")
(setf (get gazebo_radiation_plugins::MassYamlLoader :request) gazebo_radiation_plugins::MassYamlLoaderRequest)
(setf (get gazebo_radiation_plugins::MassYamlLoader :response) gazebo_radiation_plugins::MassYamlLoaderResponse)

(defmethod gazebo_radiation_plugins::MassYamlLoaderRequest
  (:response () (instance gazebo_radiation_plugins::MassYamlLoaderResponse :init)))

(setf (get gazebo_radiation_plugins::MassYamlLoaderRequest :md5sum-) "bf1b291db59c399c5c5b25859bd3e2e3")
(setf (get gazebo_radiation_plugins::MassYamlLoaderRequest :datatype-) "gazebo_radiation_plugins/MassYamlLoaderRequest")
(setf (get gazebo_radiation_plugins::MassYamlLoaderRequest :definition-)
      "string folder
---
")

(setf (get gazebo_radiation_plugins::MassYamlLoaderResponse :md5sum-) "bf1b291db59c399c5c5b25859bd3e2e3")
(setf (get gazebo_radiation_plugins::MassYamlLoaderResponse :datatype-) "gazebo_radiation_plugins/MassYamlLoaderResponse")
(setf (get gazebo_radiation_plugins::MassYamlLoaderResponse :definition-)
      "string folder
---
")



(provide :gazebo_radiation_plugins/MassYamlLoader "bf1b291db59c399c5c5b25859bd3e2e3")


