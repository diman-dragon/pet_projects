import React, { forwardRef } from "react";
import { Button, type ButtonVariant, type ButtonSize } from "../Button";

interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: ButtonVariant;
  size?: ButtonSize;
  isLoading?: boolean;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(({
  children,
  variant = "secondary",
  size = "icon",
  ...props
}, ref) => {
  return (
    <Button ref={ref} variant={variant} size={size} {...props}>
      {children}
    </Button>
  );
});

IconButton.displayName = "IconButton";
