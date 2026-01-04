
(cl:in-package :asdf)

(defsystem "gazebo_radiation_plugins-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "ConvertWorldBuilderModel" :depends-on ("_package_ConvertWorldBuilderModel"))
    (:file "_package_ConvertWorldBuilderModel" :depends-on ("_package"))
    (:file "EnvironmentEvolver" :depends-on ("_package_EnvironmentEvolver"))
    (:file "_package_EnvironmentEvolver" :depends-on ("_package"))
    (:file "GenRandomEnvironmentalEffects" :depends-on ("_package_GenRandomEnvironmentalEffects"))
    (:file "_package_GenRandomEnvironmentalEffects" :depends-on ("_package"))
    (:file "GenWorldsFromModels" :depends-on ("_package_GenWorldsFromModels"))
    (:file "_package_GenWorldsFromModels" :depends-on ("_package"))
    (:file "GenYamlsFromWorld" :depends-on ("_package_GenYamlsFromWorld"))
    (:file "_package_GenYamlsFromWorld" :depends-on ("_package"))
    (:file "MassYamlLoader" :depends-on ("_package_MassYamlLoader"))
    (:file "_package_MassYamlLoader" :depends-on ("_package"))
  ))