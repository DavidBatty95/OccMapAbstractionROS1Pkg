;; Auto-generated. Do not edit!


(when (boundp 'gazebo_radiation_plugins::EnvironmentEvolver)
  (if (not (find-package "GAZEBO_RADIATION_PLUGINS"))
    (make-package "GAZEBO_RADIATION_PLUGINS"))
  (shadow 'EnvironmentEvolver (find-package "GAZEBO_RADIATION_PLUGINS")))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::ENVIRONMENTEVOLVER")
  (make-package "GAZEBO_RADIATION_PLUGINS::ENVIRONMENTEVOLVER"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::ENVIRONMENTEVOLVERREQUEST")
  (make-package "GAZEBO_RADIATION_PLUGINS::ENVIRONMENTEVOLVERREQUEST"))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::ENVIRONMENTEVOLVERRESPONSE")
  (make-package "GAZEBO_RADIATION_PLUGINS::ENVIRONMENTEVOLVERRESPONSE"))

(in-package "ROS")





(defclass gazebo_radiation_plugins::EnvironmentEvolverRequest
  :super ros::object
  :slots ())

(defmethod gazebo_radiation_plugins::EnvironmentEvolverRequest
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

(defclass gazebo_radiation_plugins::EnvironmentEvolverResponse
  :super ros::object
  :slots (_finished ))

(defmethod gazebo_radiation_plugins::EnvironmentEvolverResponse
  (:init
   (&key
    ((:finished __finished) nil)
    )
   (send-super :init)
   (setq _finished __finished)
   self)
  (:finished
   (&optional (__finished :null))
   (if (not (eq __finished :null)) (setq _finished __finished)) _finished)
  (:serialization-length
   ()
   (+
    ;; bool _finished
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _finished
       (if _finished (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _finished
     (setq _finished (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass gazebo_radiation_plugins::EnvironmentEvolver
  :super ros::object
  :slots ())

(setf (get gazebo_radiation_plugins::EnvironmentEvolver :md5sum-) "e2797c797e620a950ba704492d72873b")
(setf (get gazebo_radiation_plugins::EnvironmentEvolver :datatype-) "gazebo_radiation_plugins/EnvironmentEvolver")
(setf (get gazebo_radiation_plugins::EnvironmentEvolver :request) gazebo_radiation_plugins::EnvironmentEvolverRequest)
(setf (get gazebo_radiation_plugins::EnvironmentEvolver :response) gazebo_radiation_plugins::EnvironmentEvolverResponse)

(defmethod gazebo_radiation_plugins::EnvironmentEvolverRequest
  (:response () (instance gazebo_radiation_plugins::EnvironmentEvolverResponse :init)))

(setf (get gazebo_radiation_plugins::EnvironmentEvolverRequest :md5sum-) "e2797c797e620a950ba704492d72873b")
(setf (get gazebo_radiation_plugins::EnvironmentEvolverRequest :datatype-) "gazebo_radiation_plugins/EnvironmentEvolverRequest")
(setf (get gazebo_radiation_plugins::EnvironmentEvolverRequest :definition-)
      "---
bool finished
")

(setf (get gazebo_radiation_plugins::EnvironmentEvolverResponse :md5sum-) "e2797c797e620a950ba704492d72873b")
(setf (get gazebo_radiation_plugins::EnvironmentEvolverResponse :datatype-) "gazebo_radiation_plugins/EnvironmentEvolverResponse")
(setf (get gazebo_radiation_plugins::EnvironmentEvolverResponse :definition-)
      "---
bool finished
")



(provide :gazebo_radiation_plugins/EnvironmentEvolver "e2797c797e620a950ba704492d72873b")


