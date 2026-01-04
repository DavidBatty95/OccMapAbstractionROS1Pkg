; Auto-generated. Do not edit!


(cl:in-package gazebo_radiation_plugins-srv)


;//! \htmlinclude GenRandomEnvironmentalEffects-request.msg.html

(cl:defclass <GenRandomEnvironmentalEffects-request> (roslisp-msg-protocol:ros-message)
  ((folder
    :reader folder
    :initarg :folder
    :type cl:string
    :initform "")
   (number_of_radiation_sources
    :reader number_of_radiation_sources
    :initarg :number_of_radiation_sources
    :type cl:fixnum
    :initform 0)
   (number_of_rusting_models
    :reader number_of_rusting_models
    :initarg :number_of_rusting_models
    :type cl:fixnum
    :initform 0)
   (number_of_bending_models
    :reader number_of_bending_models
    :initarg :number_of_bending_models
    :type cl:fixnum
    :initform 0)
   (max_bend_factor
    :reader max_bend_factor
    :initarg :max_bend_factor
    :type cl:float
    :initform 0.0)
   (max_rust_factor
    :reader max_rust_factor
    :initarg :max_rust_factor
    :type cl:float
    :initform 0.0))
)

(cl:defclass GenRandomEnvironmentalEffects-request (<GenRandomEnvironmentalEffects-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GenRandomEnvironmentalEffects-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GenRandomEnvironmentalEffects-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<GenRandomEnvironmentalEffects-request> is deprecated: use gazebo_radiation_plugins-srv:GenRandomEnvironmentalEffects-request instead.")))

(cl:ensure-generic-function 'folder-val :lambda-list '(m))
(cl:defmethod folder-val ((m <GenRandomEnvironmentalEffects-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:folder-val is deprecated.  Use gazebo_radiation_plugins-srv:folder instead.")
  (folder m))

(cl:ensure-generic-function 'number_of_radiation_sources-val :lambda-list '(m))
(cl:defmethod number_of_radiation_sources-val ((m <GenRandomEnvironmentalEffects-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:number_of_radiation_sources-val is deprecated.  Use gazebo_radiation_plugins-srv:number_of_radiation_sources instead.")
  (number_of_radiation_sources m))

(cl:ensure-generic-function 'number_of_rusting_models-val :lambda-list '(m))
(cl:defmethod number_of_rusting_models-val ((m <GenRandomEnvironmentalEffects-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:number_of_rusting_models-val is deprecated.  Use gazebo_radiation_plugins-srv:number_of_rusting_models instead.")
  (number_of_rusting_models m))

(cl:ensure-generic-function 'number_of_bending_models-val :lambda-list '(m))
(cl:defmethod number_of_bending_models-val ((m <GenRandomEnvironmentalEffects-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:number_of_bending_models-val is deprecated.  Use gazebo_radiation_plugins-srv:number_of_bending_models instead.")
  (number_of_bending_models m))

(cl:ensure-generic-function 'max_bend_factor-val :lambda-list '(m))
(cl:defmethod max_bend_factor-val ((m <GenRandomEnvironmentalEffects-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:max_bend_factor-val is deprecated.  Use gazebo_radiation_plugins-srv:max_bend_factor instead.")
  (max_bend_factor m))

(cl:ensure-generic-function 'max_rust_factor-val :lambda-list '(m))
(cl:defmethod max_rust_factor-val ((m <GenRandomEnvironmentalEffects-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:max_rust_factor-val is deprecated.  Use gazebo_radiation_plugins-srv:max_rust_factor instead.")
  (max_rust_factor m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GenRandomEnvironmentalEffects-request>) ostream)
  "Serializes a message object of type '<GenRandomEnvironmentalEffects-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'folder))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'folder))
  (cl:let* ((signed (cl:slot-value msg 'number_of_radiation_sources)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'number_of_rusting_models)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'number_of_bending_models)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'max_bend_factor))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'max_rust_factor))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GenRandomEnvironmentalEffects-request>) istream)
  "Deserializes a message object of type '<GenRandomEnvironmentalEffects-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'folder) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'folder) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'number_of_radiation_sources) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'number_of_rusting_models) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'number_of_bending_models) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'max_bend_factor) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'max_rust_factor) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GenRandomEnvironmentalEffects-request>)))
  "Returns string type for a service object of type '<GenRandomEnvironmentalEffects-request>"
  "gazebo_radiation_plugins/GenRandomEnvironmentalEffectsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GenRandomEnvironmentalEffects-request)))
  "Returns string type for a service object of type 'GenRandomEnvironmentalEffects-request"
  "gazebo_radiation_plugins/GenRandomEnvironmentalEffectsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GenRandomEnvironmentalEffects-request>)))
  "Returns md5sum for a message object of type '<GenRandomEnvironmentalEffects-request>"
  "1601fbc011dbf3b713e5f6ae36624db7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GenRandomEnvironmentalEffects-request)))
  "Returns md5sum for a message object of type 'GenRandomEnvironmentalEffects-request"
  "1601fbc011dbf3b713e5f6ae36624db7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GenRandomEnvironmentalEffects-request>)))
  "Returns full string definition for message of type '<GenRandomEnvironmentalEffects-request>"
  (cl:format cl:nil "string folder~%int16 number_of_radiation_sources~%int16 number_of_rusting_models~%int16 number_of_bending_models~%float32 max_bend_factor~%float32 max_rust_factor~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GenRandomEnvironmentalEffects-request)))
  "Returns full string definition for message of type 'GenRandomEnvironmentalEffects-request"
  (cl:format cl:nil "string folder~%int16 number_of_radiation_sources~%int16 number_of_rusting_models~%int16 number_of_bending_models~%float32 max_bend_factor~%float32 max_rust_factor~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GenRandomEnvironmentalEffects-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'folder))
     2
     2
     2
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GenRandomEnvironmentalEffects-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GenRandomEnvironmentalEffects-request
    (cl:cons ':folder (folder msg))
    (cl:cons ':number_of_radiation_sources (number_of_radiation_sources msg))
    (cl:cons ':number_of_rusting_models (number_of_rusting_models msg))
    (cl:cons ':number_of_bending_models (number_of_bending_models msg))
    (cl:cons ':max_bend_factor (max_bend_factor msg))
    (cl:cons ':max_rust_factor (max_rust_factor msg))
))
;//! \htmlinclude GenRandomEnvironmentalEffects-response.msg.html

(cl:defclass <GenRandomEnvironmentalEffects-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass GenRandomEnvironmentalEffects-response (<GenRandomEnvironmentalEffects-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GenRandomEnvironmentalEffects-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GenRandomEnvironmentalEffects-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<GenRandomEnvironmentalEffects-response> is deprecated: use gazebo_radiation_plugins-srv:GenRandomEnvironmentalEffects-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GenRandomEnvironmentalEffects-response>) ostream)
  "Serializes a message object of type '<GenRandomEnvironmentalEffects-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GenRandomEnvironmentalEffects-response>) istream)
  "Deserializes a message object of type '<GenRandomEnvironmentalEffects-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GenRandomEnvironmentalEffects-response>)))
  "Returns string type for a service object of type '<GenRandomEnvironmentalEffects-response>"
  "gazebo_radiation_plugins/GenRandomEnvironmentalEffectsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GenRandomEnvironmentalEffects-response)))
  "Returns string type for a service object of type 'GenRandomEnvironmentalEffects-response"
  "gazebo_radiation_plugins/GenRandomEnvironmentalEffectsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GenRandomEnvironmentalEffects-response>)))
  "Returns md5sum for a message object of type '<GenRandomEnvironmentalEffects-response>"
  "1601fbc011dbf3b713e5f6ae36624db7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GenRandomEnvironmentalEffects-response)))
  "Returns md5sum for a message object of type 'GenRandomEnvironmentalEffects-response"
  "1601fbc011dbf3b713e5f6ae36624db7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GenRandomEnvironmentalEffects-response>)))
  "Returns full string definition for message of type '<GenRandomEnvironmentalEffects-response>"
  (cl:format cl:nil "~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GenRandomEnvironmentalEffects-response)))
  "Returns full string definition for message of type 'GenRandomEnvironmentalEffects-response"
  (cl:format cl:nil "~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GenRandomEnvironmentalEffects-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GenRandomEnvironmentalEffects-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GenRandomEnvironmentalEffects-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GenRandomEnvironmentalEffects)))
  'GenRandomEnvironmentalEffects-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GenRandomEnvironmentalEffects)))
  'GenRandomEnvironmentalEffects-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GenRandomEnvironmentalEffects)))
  "Returns string type for a service object of type '<GenRandomEnvironmentalEffects>"
  "gazebo_radiation_plugins/GenRandomEnvironmentalEffects")