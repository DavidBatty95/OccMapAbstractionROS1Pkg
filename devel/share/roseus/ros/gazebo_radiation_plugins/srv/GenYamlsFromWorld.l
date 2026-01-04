;; Auto-generated. Do not edit!


(when (boundp 'gazebo_radiation_plugins::GenYamlsFromWorld)
  (if (not (find-package "GAZEBO_RADIATION_PLUGINS"))
    (make-package "GAZEBO_RADIATION_PLUGINS"))
  (shadow 'GenYamlsFromWorld (find-package "GAZEBO_RADIATION_PLUGINS")))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::GENYAMLSFROMWORLD")
  (make-package "GAZEBO_RADIATION_PLUGINS::GENYAMLSFROMWORLD"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::GENYAMLSFROMWORLDREQUEST")
  (make-package "GAZEBO_RADIATION_PLUGINS::GENYAMLSFROMWORLDREQUEST"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::GENYAMLSFROMWORLDRESPONSE")
  (make-package "GAZEBO_RADIATION_PLUGINS::GENYAMLSFROMWORLDRESPONSE"))

(in-package "ROS")





(defclass gazebo_radiation_plugins::GenYamlsFromWorldRequest
  :super ros::object
  :slots (_folder _world_filename _model_subset_filename _default_bend _default_rust _default_radiation ))

(defmethod gazebo_radiation_plugins::GenYamlsFromWorldRequest
  (:init
   (&key
    ((:folder __folder) "")
    ((:world_filename __world_filename) "")
    ((:model_subset_filename __model_subset_filename) "")
    ((:default_bend __default_bend) 0.0)
    ((:default_rust __default_rust) 0.0)
    ((:default_radiation __default_radiation) 0.0)
    )
   (send-super :init)
   (setq _folder (string __folder))
   (setq _world_filename (string __world_filename))
   (setq _model_subset_filename (string __model_subset_filename))
   (setq _default_bend (float __default_bend))
   (setq _default_rust (float __default_rust))
   (setq _default_radiation (float __default_radiation))
   self)
  (:folder
   (&optional __folder)
   (if __folder (setq _folder __folder)) _folder)
  (:world_filename
   (&optional __world_filename)
   (if __world_filename (setq _world_filename __world_filename)) _world_filename)
  (:model_subset_filename
   (&optional __model_subset_filename)
   (if __model_subset_filename (setq _model_subset_filename __model_subset_filename)) _model_subset_filename)
  (:default_bend
   (&optional __default_bend)
   (if __default_bend (setq _default_bend __default_bend)) _default_bend)
  (:default_rust
   (&optional __default_rust)
   (if __default_rust (setq _default_rust __default_rust)) _default_rust)
  (:default_radiation
   (&optional __default_radiation)
   (if __default_radiation (setq _default_radiation __default_radiation)) _default_radiation)
  (:serialization-length
   ()
   (+
    ;; string _folder
    4 (length _folder)
    ;; string _world_filename
    4 (length _world_filename)
    ;; string _model_subset_filename
    4 (length _model_subset_filename)
    ;; float32 _default_bend
    4
    ;; float32 _default_rust
    4
    ;; float32 _default_radiation
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _folder
       (write-long (length _folder) s) (princ _folder s)
     ;; string _world_filename
       (write-long (length _world_filename) s) (princ _world_filename s)
     ;; string _model_subset_filename
       (write-long (length _model_subset_filename) s) (princ _model_subset_filename s)
     ;; float32 _default_bend
       (sys::poke _default_bend (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _default_rust
       (sys::poke _default_rust (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _default_radiation
       (sys::poke _default_radiation (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _folder
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _folder (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _world_filename
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _world_filename (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _model_subset_filename
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _model_subset_filename (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float32 _default_bend
     (setq _default_bend (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _default_rust
     (setq _default_rust (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _default_radiation
     (setq _default_radiation (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass gazebo_radiation_plugins::GenYamlsFromWorldResponse
  :super ros::object
  :slots ())

(defmethod gazebo_radiation_plugins::GenYamlsFromWorldResponse
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

(defclass gazebo_radiation_plugins::GenYamlsFromWorld
  :super ros::object
  :slots ())

(setf (get gazebo_radiation_plugins::GenYamlsFromWorld :md5sum-) "96e84490834f1029806f056f6761f002")
(setf (get gazebo_radiation_plugins::GenYamlsFromWorld :datatype-) "gazebo_radiation_plugins/GenYamlsFromWorld")
(setf (get gazebo_radiation_plugins::GenYamlsFromWorld :request) gazebo_radiation_plugins::GenYamlsFromWorldRequest)
(setf (get gazebo_radiation_plugins::GenYamlsFromWorld :response) gazebo_radiation_plugins::GenYamlsFromWorldResponse)

(defmethod gazebo_radiation_plugins::GenYamlsFromWorldRequest
  (:response () (instance gazebo_radiation_plugins::GenYamlsFromWorldResponse :init)))

(setf (get gazebo_radiation_plugins::GenYamlsFromWorldRequest :md5sum-) "96e84490834f1029806f056f6761f002")
(setf (get gazebo_radiation_plugins::GenYamlsFromWorldRequest :datatype-) "gazebo_radiation_plugins/GenYamlsFromWorldRequest")
(setf (get gazebo_radiation_plugins::GenYamlsFromWorldRequest :definition-)
      "string folder
string world_filename
string model_subset_filename
float32 default_bend
float32 default_rust
float32 default_radiation
---


")

(setf (get gazebo_radiation_plugins::GenYamlsFromWorldResponse :md5sum-) "96e84490834f1029806f056f6761f002")
(setf (get gazebo_radiation_plugins::GenYamlsFromWorldResponse :datatype-) "gazebo_radiation_plugins/GenYamlsFromWorldResponse")
(setf (get gazebo_radiation_plugins::GenYamlsFromWorldResponse :definition-)
      "string folder
string world_filename
string model_subset_filename
float32 default_bend
float32 default_rust
float32 default_radiation
---


")



(provide :gazebo_radiation_plugins/GenYamlsFromWorld "96e84490834f1029806f056f6761f002")


