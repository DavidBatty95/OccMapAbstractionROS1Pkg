; Auto-generated. Do not edit!


(cl:in-package gazebo_radiation_plugins-srv)


;//! \htmlinclude GenYamlsFromWorld-request.msg.html

(cl:defclass <GenYamlsFromWorld-request> (roslisp-msg-protocol:ros-message)
  ((folder
    :reader folder
    :initarg :folder
    :type cl:string
    :initform "")
   (world_filename
    :reader world_filename
    :initarg :world_filename
    :type cl:string
    :initform "")
   (model_subset_filename
    :reader model_subset_filename
    :initarg :model_subset_filename
    :type cl:string
    :initform "")
   (default_bend
    :reader default_bend
    :initarg :default_bend
    :type cl:float
    :initform 0.0)
   (default_rust
    :reader default_rust
    :initarg :default_rust
    :type cl:float
    :initform 0.0)
   (default_radiation
    :reader default_radiation
    :initarg :default_radiation
    :type cl:float
    :initform 0.0))
)

(cl:defclass GenYamlsFromWorld-request (<GenYamlsFromWorld-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GenYamlsFromWorld-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GenYamlsFromWorld-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<GenYamlsFromWorld-request> is deprecated: use gazebo_radiation_plugins-srv:GenYamlsFromWorld-request instead.")))

(cl:ensure-generic-function 'folder-val :lambda-list '(m))
(cl:defmethod folder-val ((m <GenYamlsFromWorld-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:folder-val is deprecated.  Use gazebo_radiation_plugins-srv:folder instead.")
  (folder m))

(cl:ensure-generic-function 'world_filename-val :lambda-list '(m))
(cl:defmethod world_filename-val ((m <GenYamlsFromWorld-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:world_filename-val is deprecated.  Use gazebo_radiation_plugins-srv:world_filename instead.")
  (world_filename m))

(cl:ensure-generic-function 'model_subset_filename-val :lambda-list '(m))
(cl:defmethod model_subset_filename-val ((m <GenYamlsFromWorld-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:model_subset_filename-val is deprecated.  Use gazebo_radiation_plugins-srv:model_subset_filename instead.")
  (model_subset_filename m))

(cl:ensure-generic-function 'default_bend-val :lambda-list '(m))
(cl:defmethod default_bend-val ((m <GenYamlsFromWorld-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:default_bend-val is deprecated.  Use gazebo_radiation_plugins-srv:default_bend instead.")
  (default_bend m))

(cl:ensure-generic-function 'default_rust-val :lambda-list '(m))
(cl:defmethod default_rust-val ((m <GenYamlsFromWorld-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:default_rust-val is deprecated.  Use gazebo_radiation_plugins-srv:default_rust instead.")
  (default_rust m))

(cl:ensure-generic-function 'default_radiation-val :lambda-list '(m))
(cl:defmethod default_radiation-val ((m <GenYamlsFromWorld-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:default_radiation-val is deprecated.  Use gazebo_radiation_plugins-srv:default_radiation instead.")
  (default_radiation m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GenYamlsFromWorld-request>) ostream)
  "Serializes a message object of type '<GenYamlsFromWorld-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'folder))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'folder))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'world_filename))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'world_filename))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'model_subset_filename))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'model_subset_filename))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'default_bend))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'default_rust))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'default_radiation))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GenYamlsFromWorld-request>) istream)
  "Deserializes a message object of type '<GenYamlsFromWorld-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'folder) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'folder) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'world_filename) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'world_filename) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'model_subset_filename) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'model_subset_filename) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'default_bend) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'default_rust) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'default_radiation) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GenYamlsFromWorld-request>)))
  "Returns string type for a service object of type '<GenYamlsFromWorld-request>"
  "gazebo_radiation_plugins/GenYamlsFromWorldRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GenYamlsFromWorld-request)))
  "Returns string type for a service object of type 'GenYamlsFromWorld-request"
  "gazebo_radiation_plugins/GenYamlsFromWorldRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GenYamlsFromWorld-request>)))
  "Returns md5sum for a message object of type '<GenYamlsFromWorld-request>"
  "96e84490834f1029806f056f6761f002")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GenYamlsFromWorld-request)))
  "Returns md5sum for a message object of type 'GenYamlsFromWorld-request"
  "96e84490834f1029806f056f6761f002")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GenYamlsFromWorld-request>)))
  "Returns full string definition for message of type '<GenYamlsFromWorld-request>"
  (cl:format cl:nil "string folder~%string world_filename~%string model_subset_filename~%float32 default_bend~%float32 default_rust~%float32 default_radiation~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GenYamlsFromWorld-request)))
  "Returns full string definition for message of type 'GenYamlsFromWorld-request"
  (cl:format cl:nil "string folder~%string world_filename~%string model_subset_filename~%float32 default_bend~%float32 default_rust~%float32 default_radiation~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GenYamlsFromWorld-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'folder))
     4 (cl:length (cl:slot-value msg 'world_filename))
     4 (cl:length (cl:slot-value msg 'model_subset_filename))
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GenYamlsFromWorld-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GenYamlsFromWorld-request
    (cl:cons ':folder (folder msg))
    (cl:cons ':world_filename (world_filename msg))
    (cl:cons ':model_subset_filename (model_subset_filename msg))
    (cl:cons ':default_bend (default_bend msg))
    (cl:cons ':default_rust (default_rust msg))
    (cl:cons ':default_radiation (default_radiation msg))
))
;//! \htmlinclude GenYamlsFromWorld-response.msg.html

(cl:defclass <GenYamlsFromWorld-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass GenYamlsFromWorld-response (<GenYamlsFromWorld-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GenYamlsFromWorld-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GenYamlsFromWorld-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<GenYamlsFromWorld-response> is deprecated: use gazebo_radiation_plugins-srv:GenYamlsFromWorld-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GenYamlsFromWorld-response>) ostream)
  "Serializes a message object of type '<GenYamlsFromWorld-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GenYamlsFromWorld-response>) istream)
  "Deserializes a message object of type '<GenYamlsFromWorld-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GenYamlsFromWorld-response>)))
  "Returns string type for a service object of type '<GenYamlsFromWorld-response>"
  "gazebo_radiation_plugins/GenYamlsFromWorldResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GenYamlsFromWorld-response)))
  "Returns string type for a service object of type 'GenYamlsFromWorld-response"
  "gazebo_radiation_plugins/GenYamlsFromWorldResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GenYamlsFromWorld-response>)))
  "Returns md5sum for a message object of type '<GenYamlsFromWorld-response>"
  "96e84490834f1029806f056f6761f002")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GenYamlsFromWorld-response)))
  "Returns md5sum for a message object of type 'GenYamlsFromWorld-response"
  "96e84490834f1029806f056f6761f002")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GenYamlsFromWorld-response>)))
  "Returns full string definition for message of type '<GenYamlsFromWorld-response>"
  (cl:format cl:nil "~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GenYamlsFromWorld-response)))
  "Returns full string definition for message of type 'GenYamlsFromWorld-response"
  (cl:format cl:nil "~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GenYamlsFromWorld-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GenYamlsFromWorld-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GenYamlsFromWorld-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GenYamlsFromWorld)))
  'GenYamlsFromWorld-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GenYamlsFromWorld)))
  'GenYamlsFromWorld-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GenYamlsFromWorld)))
  "Returns string type for a service object of type '<GenYamlsFromWorld>"
  "gazebo_radiation_plugins/GenYamlsFromWorld")