;; Auto-generated. Do not edit!


(when (boundp 'gazebo_radiation_plugins::GenWorldsFromModels)
  (if (not (find-package "GAZEBO_RADIATION_PLUGINS"))
    (make-package "GAZEBO_RADIATION_PLUGINS"))
  (shadow 'GenWorldsFromModels (find-package "GAZEBO_RADIATION_PLUGINS")))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::GENWORLDSFROMMODELS")
  (make-package "GAZEBO_RADIATION_PLUGINS::GENWORLDSFROMMODELS"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::GENWORLDSFROMMODELSREQUEST")
  (make-package "GAZEBO_RADIATION_PLUGINS::GENWORLDSFROMMODELSREQUEST"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::GENWORLDSFROMMODELSRESPONSE")
  (make-package "GAZEBO_RADIATION_PLUGINS::GENWORLDSFROMMODELSRESPONSE"))

(in-package "ROS")





(defclass gazebo_radiation_plugins::GenWorldsFromModelsRequest
  :super ros::object
  :slots (_folder _model_filename ))

(defmethod gazebo_radiation_plugins::GenWorldsFromModelsRequest
  (:init
   (&key
    ((:folder __folder) "")
    ((:model_filename __model_filename) "")
    )
   (send-super :init)
   (setq _folder (string __folder))
   (setq _model_filename (string __model_filename))
   self)
  (:folder
   (&optional __folder)
   (if __folder (setq _folder __folder)) _folder)
  (:model_filename
   (&optional __model_filename)
   (if __model_filename (setq _model_filename __model_filename)) _model_filename)
  (:serialization-length
   ()
   (+
    ;; string _folder
    4 (length _folder)
    ;; string _model_filename
    4 (length _model_filename)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _folder
       (write-long (length _folder) s) (princ _folder s)
     ;; string _model_filename
       (write-long (length _model_filename) s) (princ _model_filename s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _folder
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _folder (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _model_filename
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _model_filename (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass gazebo_radiation_plugins::GenWorldsFromModelsResponse
  :super ros::object
  :slots ())

(defmethod gazebo_radiation_plugins::GenWorldsFromModelsResponse
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

(defclass gazebo_radiation_plugins::GenWorldsFromModels
  :super ros::object
  :slots ())

(setf (get gazebo_radiation_plugins::GenWorldsFromModels :md5sum-) "aab9442741bfdbb3be9f5e426f1408ca")
(setf (get gazebo_radiation_plugins::GenWorldsFromModels :datatype-) "gazebo_radiation_plugins/GenWorldsFromModels")
(setf (get gazebo_radiation_plugins::GenWorldsFromModels :request) gazebo_radiation_plugins::GenWorldsFromModelsRequest)
(setf (get gazebo_radiation_plugins::GenWorldsFromModels :response) gazebo_radiation_plugins::GenWorldsFromModelsResponse)

(defmethod gazebo_radiation_plugins::GenWorldsFromModelsRequest
  (:response () (instance gazebo_radiation_plugins::GenWorldsFromModelsResponse :init)))

(setf (get gazebo_radiation_plugins::GenWorldsFromModelsRequest :md5sum-) "aab9442741bfdbb3be9f5e426f1408ca")
(setf (get gazebo_radiation_plugins::GenWorldsFromModelsRequest :datatype-) "gazebo_radiation_plugins/GenWorldsFromModelsRequest")
(setf (get gazebo_radiation_plugins::GenWorldsFromModelsRequest :definition-)
      "string folder
string model_filename
---


")

(setf (get gazebo_radiation_plugins::GenWorldsFromModelsResponse :md5sum-) "aab9442741bfdbb3be9f5e426f1408ca")
(setf (get gazebo_radiation_plugins::GenWorldsFromModelsResponse :datatype-) "gazebo_radiation_plugins/GenWorldsFromModelsResponse")
(setf (get gazebo_radiation_plugins::GenWorldsFromModelsResponse :definition-)
      "string folder
string model_filename
---


")



(provide :gazebo_radiation_plugins/GenWorldsFromModels "aab9442741bfdbb3be9f5e426f1408ca")


