; Auto-generated. Do not edit!


(cl:in-package gazebo_radiation_plugins-srv)


;//! \htmlinclude EnvironmentEvolver-request.msg.html

(cl:defclass <EnvironmentEvolver-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass EnvironmentEvolver-request (<EnvironmentEvolver-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EnvironmentEvolver-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EnvironmentEvolver-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<EnvironmentEvolver-request> is deprecated: use gazebo_radiation_plugins-srv:EnvironmentEvolver-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EnvironmentEvolver-request>) ostream)
  "Serializes a message object of type '<EnvironmentEvolver-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EnvironmentEvolver-request>) istream)
  "Deserializes a message object of type '<EnvironmentEvolver-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EnvironmentEvolver-request>)))
  "Returns string type for a service object of type '<EnvironmentEvolver-request>"
  "gazebo_radiation_plugins/EnvironmentEvolverRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EnvironmentEvolver-request)))
  "Returns string type for a service object of type 'EnvironmentEvolver-request"
  "gazebo_radiation_plugins/EnvironmentEvolverRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EnvironmentEvolver-request>)))
  "Returns md5sum for a message object of type '<EnvironmentEvolver-request>"
  "e2797c797e620a950ba704492d72873b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EnvironmentEvolver-request)))
  "Returns md5sum for a message object of type 'EnvironmentEvolver-request"
  "e2797c797e620a950ba704492d72873b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EnvironmentEvolver-request>)))
  "Returns full string definition for message of type '<EnvironmentEvolver-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EnvironmentEvolver-request)))
  "Returns full string definition for message of type 'EnvironmentEvolver-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EnvironmentEvolver-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EnvironmentEvolver-request>))
  "Converts a ROS message object to a list"
  (cl:list 'EnvironmentEvolver-request
))
;//! \htmlinclude EnvironmentEvolver-response.msg.html

(cl:defclass <EnvironmentEvolver-response> (roslisp-msg-protocol:ros-message)
  ((finished
    :reader finished
    :initarg :finished
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass EnvironmentEvolver-response (<EnvironmentEvolver-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EnvironmentEvolver-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EnvironmentEvolver-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<EnvironmentEvolver-response> is deprecated: use gazebo_radiation_plugins-srv:EnvironmentEvolver-response instead.")))

(cl:ensure-generic-function 'finished-val :lambda-list '(m))
(cl:defmethod finished-val ((m <EnvironmentEvolver-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:finished-val is deprecated.  Use gazebo_radiation_plugins-srv:finished instead.")
  (finished m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EnvironmentEvolver-response>) ostream)
  "Serializes a message object of type '<EnvironmentEvolver-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'finished) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EnvironmentEvolver-response>) istream)
  "Deserializes a message object of type '<EnvironmentEvolver-response>"
    (cl:setf (cl:slot-value msg 'finished) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EnvironmentEvolver-response>)))
  "Returns string type for a service object of type '<EnvironmentEvolver-response>"
  "gazebo_radiation_plugins/EnvironmentEvolverResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EnvironmentEvolver-response)))
  "Returns string type for a service object of type 'EnvironmentEvolver-response"
  "gazebo_radiation_plugins/EnvironmentEvolverResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EnvironmentEvolver-response>)))
  "Returns md5sum for a message object of type '<EnvironmentEvolver-response>"
  "e2797c797e620a950ba704492d72873b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EnvironmentEvolver-response)))
  "Returns md5sum for a message object of type 'EnvironmentEvolver-response"
  "e2797c797e620a950ba704492d72873b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EnvironmentEvolver-response>)))
  "Returns full string definition for message of type '<EnvironmentEvolver-response>"
  (cl:format cl:nil "bool finished~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EnvironmentEvolver-response)))
  "Returns full string definition for message of type 'EnvironmentEvolver-response"
  (cl:format cl:nil "bool finished~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EnvironmentEvolver-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EnvironmentEvolver-response>))
  "Converts a ROS message object to a list"
  (cl:list 'EnvironmentEvolver-response
    (cl:cons ':finished (finished msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'EnvironmentEvolver)))
  'EnvironmentEvolver-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'EnvironmentEvolver)))
  'EnvironmentEvolver-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EnvironmentEvolver)))
  "Returns string type for a service object of type '<EnvironmentEvolver>"
  "gazebo_radiation_plugins/EnvironmentEvolver")