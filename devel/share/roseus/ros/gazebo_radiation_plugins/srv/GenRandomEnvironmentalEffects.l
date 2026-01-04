;; Auto-generated. Do not edit!


(when (boundp 'gazebo_radiation_plugins::GenRandomEnvironmentalEffects)
  (if (not (find-package "GAZEBO_RADIATION_PLUGINS"))
    (make-package "GAZEBO_RADIATION_PLUGINS"))
  (shadow 'GenRandomEnvironmentalEffects (find-package "GAZEBO_RADIATION_PLUGINS")))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::GENRANDOMENVIRONMENTALEFFECTS")
  (make-package "GAZEBO_RADIATION_PLUGINS::GENRANDOMENVIRONMENTALEFFECTS"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::GENRANDOMENVIRONMENTALEFFECTSREQUEST")
  (make-package "GAZEBO_RADIATION_PLUGINS::GENRANDOMENVIRONMENTALEFFECTSREQUEST"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::GENRANDOMENVIRONMENTALEFFECTSRESPONSE")
  (make-package "GAZEBO_RADIATION_PLUGINS::GENRANDOMENVIRONMENTALEFFECTSRESPONSE"))

(in-package "ROS")





(defclass gazebo_radiation_plugins::GenRandomEnvironmentalEffectsRequest
  :super ros::object
  :slots (_folder _number_of_radiation_sources _number_of_rusting_models _number_of_bending_models _max_bend_factor _max_rust_factor ))

(defmethod gazebo_radiation_plugins::GenRandomEnvironmentalEffectsRequest
  (:init
   (&key
    ((:folder __folder) "")
    ((:number_of_radiation_sources __number_of_radiation_sources) 0)
    ((:number_of_rusting_models __number_of_rusting_models) 0)
    ((:number_of_bending_models __number_of_bending_models) 0)
    ((:max_bend_factor __max_bend_factor) 0.0)
    ((:max_rust_factor __max_rust_factor) 0.0)
    )
   (send-super :init)
   (setq _folder (string __folder))
   (setq _number_of_radiation_sources (round __number_of_radiation_sources))
   (setq _number_of_rusting_models (round __number_of_rusting_models))
   (setq _number_of_bending_models (round __number_of_bending_models))
   (setq _max_bend_factor (float __max_bend_factor))
   (setq _max_rust_factor (float __max_rust_factor))
   self)
  (:folder
   (&optional __folder)
   (if __folder (setq _folder __folder)) _folder)
  (:number_of_radiation_sources
   (&optional __number_of_radiation_sources)
   (if __number_of_radiation_sources (setq _number_of_radiation_sources __number_of_radiation_sources)) _number_of_radiation_sources)
  (:number_of_rusting_models
   (&optional __number_of_rusting_models)
   (if __number_of_rusting_models (setq _number_of_rusting_models __number_of_rusting_models)) _number_of_rusting_models)
  (:number_of_bending_models
   (&optional __number_of_bending_models)
   (if __number_of_bending_models (setq _number_of_bending_models __number_of_bending_models)) _number_of_bending_models)
  (:max_bend_factor
   (&optional __max_bend_factor)
   (if __max_bend_factor (setq _max_bend_factor __max_bend_factor)) _max_bend_factor)
  (:max_rust_factor
   (&optional __max_rust_factor)
   (if __max_rust_factor (setq _max_rust_factor __max_rust_factor)) _max_rust_factor)
  (:serialization-length
   ()
   (+
    ;; string _folder
    4 (length _folder)
    ;; int16 _number_of_radiation_sources
    2
    ;; int16 _number_of_rusting_models
    2
    ;; int16 _number_of_bending_models
    2
    ;; float32 _max_bend_factor
    4
    ;; float32 _max_rust_factor
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _folder
       (write-long (length _folder) s) (princ _folder s)
     ;; int16 _number_of_radiation_sources
       (write-word _number_of_radiation_sources s)
     ;; int16 _number_of_rusting_models
       (write-word _number_of_rusting_models s)
     ;; int16 _number_of_bending_models
       (write-word _number_of_bending_models s)
     ;; float32 _max_bend_factor
       (sys::poke _max_bend_factor (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _max_rust_factor
       (sys::poke _max_rust_factor (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _folder
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _folder (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; int16 _number_of_radiation_sources
     (setq _number_of_radiation_sources (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; int16 _number_of_rusting_models
     (setq _number_of_rusting_models (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; int16 _number_of_bending_models
     (setq _number_of_bending_models (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; float32 _max_bend_factor
     (setq _max_bend_factor (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _max_rust_factor
     (setq _max_rust_factor (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass gazebo_radiation_plugins::GenRandomEnvironmentalEffectsResponse
  :super ros::object
  :slots ())

(defmethod gazebo_radiation_plugins::GenRandomEnvironmentalEffectsResponse
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

(defclass gazebo_radiation_plugins::GenRandomEnvironmentalEffects
  :super ros::object
  :slots ())

(setf (get gazebo_radiation_plugins::GenRandomEnvironmentalEffects :md5sum-) "1601fbc011dbf3b713e5f6ae36624db7")
(setf (get gazebo_radiation_plugins::GenRandomEnvironmentalEffects :datatype-) "gazebo_radiation_plugins/GenRandomEnvironmentalEffects")
(setf (get gazebo_radiation_plugins::GenRandomEnvironmentalEffects :request) gazebo_radiation_plugins::GenRandomEnvironmentalEffectsRequest)
(setf (get gazebo_radiation_plugins::GenRandomEnvironmentalEffects :response) gazebo_radiation_plugins::GenRandomEnvironmentalEffectsResponse)

(defmethod gazebo_radiation_plugins::GenRandomEnvironmentalEffectsRequest
  (:response () (instance gazebo_radiation_plugins::GenRandomEnvironmentalEffectsResponse :init)))

(setf (get gazebo_radiation_plugins::GenRandomEnvironmentalEffectsRequest :md5sum-) "1601fbc011dbf3b713e5f6ae36624db7")
(setf (get gazebo_radiation_plugins::GenRandomEnvironmentalEffectsRequest :datatype-) "gazebo_radiation_plugins/GenRandomEnvironmentalEffectsRequest")
(setf (get gazebo_radiation_plugins::GenRandomEnvironmentalEffectsRequest :definition-)
      "string folder
int16 number_of_radiation_sources
int16 number_of_rusting_models
int16 number_of_bending_models
float32 max_bend_factor
float32 max_rust_factor
---


")

(setf (get gazebo_radiation_plugins::GenRandomEnvironmentalEffectsResponse :md5sum-) "1601fbc011dbf3b713e5f6ae36624db7")
(setf (get gazebo_radiation_plugins::GenRandomEnvironmentalEffectsResponse :datatype-) "gazebo_radiation_plugins/GenRandomEnvironmentalEffectsResponse")
(setf (get gazebo_radiation_plugins::GenRandomEnvironmentalEffectsResponse :definition-)
      "string folder
int16 number_of_radiation_sources
int16 number_of_rusting_models
int16 number_of_bending_models
float32 max_bend_factor
float32 max_rust_factor
---


")



(provide :gazebo_radiation_plugins/GenRandomEnvironmentalEffects "1601fbc011dbf3b713e5f6ae36624db7")


