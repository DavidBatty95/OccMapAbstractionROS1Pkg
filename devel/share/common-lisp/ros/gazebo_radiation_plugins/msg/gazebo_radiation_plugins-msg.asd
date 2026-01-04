
(cl:in-package :asdf)

(defsystem "gazebo_radiation_plugins-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "Simulated_Radiation_Msg" :depends-on ("_package_Simulated_Radiation_Msg"))
    (:file "_package_Simulated_Radiation_Msg" :depends-on ("_package"))
  ))