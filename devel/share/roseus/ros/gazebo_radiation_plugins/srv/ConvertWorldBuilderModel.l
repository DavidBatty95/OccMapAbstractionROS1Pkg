;; Auto-generated. Do not edit!


(when (boundp 'gazebo_radiation_plugins::ConvertWorldBuilderModel)
  (if (not (find-package "GAZEBO_RADIATION_PLUGINS"))
    (make-package "GAZEBO_RADIATION_PLUGINS"))
  (shadow 'ConvertWorldBuilderModel (find-package "GAZEBO_RADIATION_PLUGINS")))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::CONVERTWORLDBUILDERMODEL")
  (make-package "GAZEBO_RADIATION_PLUGINS::CONVERTWORLDBUILDERMODEL"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::CONVERTWORLDBUILDERMODELREQUEST")
  (make-package "GAZEBO_RADIATION_PLUGINS::CONVERTWORLDBUILDERMODELREQUEST"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::CONVERTWORLDBUILDERMODELRESPONSE")
  (make-package "GAZEBO_RADIATION_PLUGINS::CONVERTWORLDBUILDERMODELRESPONSE"))

(in-package "ROS")





(defclass gazebo_radiation_plugins::ConvertWorldBuilderModelRequest
  :super ros::object
  :slots (_world_builder_file _gazebo_model_path _noise_file ))

(defmethod gazebo_radiation_plugins::ConvertWorldBuilderModelRequest
  (:init
   (&key
    ((:world_builder_file __world_builder_file) "")
    ((:gazebo_model_path __gazebo_model_path) "")
    ((:noise_file __noise_file) "")
    )
   (send-super :init)
   (setq _world_builder_file (string __world_builder_file))
   (setq _gazebo_model_path (string __gazebo_model_path))
   (setq _noise_file (string __noise_file))
   self)
  (:world_builder_file
   (&optional __world_builder_file)
   (if __world_builder_file (setq _world_builder_file __world_builder_file)) _world_builder_file)
  (:gazebo_model_path
   (&optional __gazebo_model_path)
   (if __gazebo_model_path (setq _gazebo_model_path __gazebo_model_path)) _gazebo_model_path)
  (:noise_file
   (&optional __noise_file)
   (if __noise_file (setq _noise_file __noise_file)) _noise_file)
  (:serialization-length
   ()
   (+
    ;; string _world_builder_file
    4 (length _world_builder_file)
    ;; string _gazebo_model_path
    4 (length _gazebo_model_path)
    ;; string _noise_file
    4 (length _noise_file)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _world_builder_file
       (write-long (length _world_builder_file) s) (princ _world_builder_file s)
     ;; string _gazebo_model_path
       (write-long (length _gazebo_model_path) s) (princ _gazebo_model_path s)
     ;; string _noise_file
       (write-long (length _noise_file) s) (princ _noise_file s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _world_builder_file
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _world_builder_file (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _gazebo_model_path
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _gazebo_model_path (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _noise_file
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _noise_file (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass gazebo_radiation_plugins::ConvertWorldBuilderModelResponse
  :super ros::object
  :slots ())

(defmethod gazebo_radiation_plugins::ConvertWorldBuilderModelResponse
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

(defclass gazebo_radiation_plugins::ConvertWorldBuilderModel
  :super ros::object
  :slots ())

(setf (get gazebo_radiation_plugins::ConvertWorldBuilderModel :md5sum-) "ca5346b14c3e8fe8c1488efbdb6aac5b")
(setf (get gazebo_radiation_plugins::ConvertWorldBuilderModel :datatype-) "gazebo_radiation_plugins/ConvertWorldBuilderModel")
(setf (get gazebo_radiation_plugins::ConvertWorldBuilderModel :request) gazebo_radiation_plugins::ConvertWorldBuilderModelRequest)
(setf (get gazebo_radiation_plugins::ConvertWorldBuilderModel :response) gazebo_radiation_plugins::ConvertWorldBuilderModelResponse)

(defmethod gazebo_radiation_plugins::ConvertWorldBuilderModelRequest
  (:response () (instance gazebo_radiation_plugins::ConvertWorldBuilderModelResponse :init)))

(setf (get gazebo_radiation_plugins::ConvertWorldBuilderModelRequest :md5sum-) "ca5346b14c3e8fe8c1488efbdb6aac5b")
(setf (get gazebo_radiation_plugins::ConvertWorldBuilderModelRequest :datatype-) "gazebo_radiation_plugins/ConvertWorldBuilderModelRequest")
(setf (get gazebo_radiation_plugins::ConvertWorldBuilderModelRequest :definition-)
      "string world_builder_file
string gazebo_model_path
string noise_file
---
")

(setf (get gazebo_radiation_plugins::ConvertWorldBuilderModelResponse :md5sum-) "ca5346b14c3e8fe8c1488efbdb6aac5b")
(setf (get gazebo_radiation_plugins::ConvertWorldBuilderModelResponse :datatype-) "gazebo_radiation_plugins/ConvertWorldBuilderModelResponse")
(setf (get gazebo_radiation_plugins::ConvertWorldBuilderModelResponse :definition-)
      "string world_builder_file
string gazebo_model_path
string noise_file
---
")



(provide :gazebo_radiation_plugins/ConvertWorldBuilderModel "ca5346b14c3e8fe8c1488efbdb6aac5b")


